import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline import (
    BaselineDataset,
    ConvSpeakerEncoder,
    Disentangler,
    LogMelFrontend,
    TripletEnvironmentDataset,
    build_manifest,
    create_label_maps,
    ensure_dir,
    grad_reverse,
    load_config,
    mapc_loss,
    save_json,
    seed_everything,
    split_manifest,
    triplet_distance_loss,
)


def train_baseline(config, splits, speaker_to_idx, run_dir, device):
    frontend = LogMelFrontend(config["sample_rate"], config["n_mels"]).to(device)
    encoder = ConvSpeakerEncoder(config["n_mels"], config["embedding_dim"]).to(device)
    classifier = nn.Linear(config["embedding_dim"], len(speaker_to_idx)).to(device)

    loader = DataLoader(
        BaselineDataset(
            splits["train"],
            speaker_to_idx,
            sample_rate=config["sample_rate"],
            segment_seconds=config["segment_seconds"],
            training=True,
        ),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(
        list(frontend.parameters()) + list(encoder.parameters()) + list(classifier.parameters()),
        lr=config["lr"],
    )
    criterion = nn.CrossEntropyLoss()
    history = []
    best_loss = float("inf")
    checkpoint_path = run_dir / "best.pt"

    for epoch in range(config["epochs"]):
        frontend.train()
        encoder.train()
        classifier.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0

        for batch in loader:
            wave = batch["waveform"].to(device)
            labels = batch["speaker"].to(device)
            feats = frontend(wave).squeeze(1)
            emb = encoder(feats)
            logits = classifier(emb)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_items += labels.size(0)

        epoch_loss = total_loss / max(1, total_items)
        epoch_acc = total_correct / max(1, total_items)
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "acc": epoch_acc})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "frontend": frontend.state_dict(),
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "history": history,
                    "config": config,
                    "model_type": "baseline",
                },
                checkpoint_path,
            )
        print(f"epoch {epoch + 1}/{config['epochs']} loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

    return checkpoint_path, history


def train_disentangled(config, splits, speaker_to_idx, run_dir, device):
    frontend = LogMelFrontend(config["sample_rate"], config["n_mels"]).to(device)
    encoder = ConvSpeakerEncoder(config["n_mels"], config["embedding_dim"]).to(device)
    disentangler = Disentangler(
        embedding_dim=config["embedding_dim"],
        latent_dim=config["latent_dim"],
        n_speakers=len(speaker_to_idx),
        n_envs=len(config["env_names"]),
    ).to(device)

    loader = DataLoader(
        TripletEnvironmentDataset(
            splits["train"],
            speaker_to_idx,
            sample_rate=config["sample_rate"],
            segment_seconds=config["segment_seconds"],
            steps_per_epoch=config["steps_per_epoch"],
        ),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    adv_params = list(disentangler.adv_env_head.parameters())
    main_params = list(frontend.parameters()) + list(encoder.parameters())
    main_params += [
        param
        for name, param in disentangler.named_parameters()
        if not name.startswith("adv_env_head.")
    ]
    main_optimizer = torch.optim.Adam(main_params, lr=config["lr"])
    adv_optimizer = torch.optim.Adam(adv_params, lr=config["lr"])
    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()

    history = []
    best_loss = float("inf")
    checkpoint_path = run_dir / "best.pt"

    for epoch in range(config["epochs"]):
        frontend.train()
        encoder.train()
        disentangler.train()
        total_loss = 0.0
        steps = 0
        last_imp = 0.0

        for batch in loader:
            waveforms = batch["waveforms"].to(device)
            labels = batch["speaker"].to(device)
            env_ids = batch["env_ids"].to(device)
            batch_size = waveforms.shape[0]

            flat_wave = waveforms.view(-1, waveforms.shape[-2], waveforms.shape[-1])
            feats = frontend(flat_wave).squeeze(1)
            embeddings = encoder(feats)
            spk_code, env_code, recon = disentangler.encode(embeddings)

            embeddings = embeddings.view(batch_size, 3, -1)
            spk_code = spk_code.view(batch_size, 3, -1)
            env_code = env_code.view(batch_size, 3, -1)
            recon = recon.view(batch_size, 3, -1)

            adv_anchor = disentangler.adv_env_head(spk_code[:, 0, :].detach())
            adv_pos = disentangler.adv_env_head(spk_code[:, 1, :].detach())
            adv_neg = disentangler.adv_env_head(spk_code[:, 2, :].detach())
            adv_disc_loss = triplet_distance_loss(adv_anchor, adv_pos, adv_neg, config["margin"])
            adv_optimizer.zero_grad()
            adv_disc_loss.backward()
            adv_optimizer.step()

            speaker_logits = disentangler.spk_classifier(spk_code[:, 0, :])
            speaker_loss = ce(speaker_logits, labels)
            recon_loss = l1(recon, embeddings)

            env_anchor = disentangler.env_head(env_code[:, 0, :])
            env_pos = disentangler.env_head(env_code[:, 1, :])
            env_neg = disentangler.env_head(env_code[:, 2, :])
            env_loss = triplet_distance_loss(env_anchor, env_pos, env_neg, config["margin"])

            rev_anchor = disentangler.adv_env_head(grad_reverse(spk_code[:, 0, :], config["lambda_adv"]))
            rev_pos = disentangler.adv_env_head(grad_reverse(spk_code[:, 1, :], config["lambda_adv"]))
            rev_neg = disentangler.adv_env_head(grad_reverse(spk_code[:, 2, :], config["lambda_adv"]))
            adv_loss = triplet_distance_loss(rev_anchor, rev_pos, rev_neg, config["margin"])

            corr_loss = mapc_loss(spk_code.reshape(-1, spk_code.shape[-1]), env_code.reshape(-1, env_code.shape[-1]))

            total = (
                config["lambda_spk"] * speaker_loss
                + config["lambda_recon"] * recon_loss
                + config["lambda_env"] * env_loss
                + adv_loss
                + config["lambda_corr"] * corr_loss
            )

            if config["model_type"] == "improved":
                env_logits = disentangler.env_classifier(env_code.reshape(-1, env_code.shape[-1]))
                improvement_loss = ce(env_logits, env_ids.reshape(-1))
                total = total + config["lambda_env_ce"] * improvement_loss
                last_imp = float(improvement_loss.item())

            main_optimizer.zero_grad()
            total.backward()
            main_optimizer.step()

            total_loss += total.item()
            steps += 1

        epoch_loss = total_loss / max(1, steps)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "frontend": frontend.state_dict(),
                    "encoder": encoder.state_dict(),
                    "disentangler": disentangler.state_dict(),
                    "history": history,
                    "config": config,
                    "model_type": config["model_type"],
                },
                checkpoint_path,
            )
        print(f"epoch {epoch + 1}/{config['epochs']} loss={epoch_loss:.4f} imp={last_imp:.4f}")

    return checkpoint_path, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    seed_everything(config["seed"])

    q2_dir = Path(__file__).resolve().parent
    dataset_root = q2_dir / config["dataset"]["root"]
    manifest_path = q2_dir / "results" / "tables" / "manifest.json"
    split_path = q2_dir / "results" / "tables" / "splits.json"

    rows = build_manifest(dataset_root, manifest_path)
    splits = split_manifest(rows, split_path, config["train_ratio"], config["val_ratio"], config["seed"])
    speaker_to_idx, chapter_to_idx = create_label_maps(rows)

    run_dir = q2_dir / "results" / "checkpoints" / config["run_name"]
    ensure_dir(run_dir)
    save_json({"speaker_to_idx": speaker_to_idx, "chapter_to_idx": chapter_to_idx}, run_dir / "label_maps.json")
    save_json({"train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"])}, run_dir / "split_sizes.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["model_type"] == "baseline":
        checkpoint_path, history = train_baseline(config, splits, speaker_to_idx, run_dir, device)
    else:
        checkpoint_path, history = train_disentangled(config, splits, speaker_to_idx, run_dir, device)

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
