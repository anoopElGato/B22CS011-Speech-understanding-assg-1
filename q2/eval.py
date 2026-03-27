import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pipeline import (
    ENVIRONMENTS,
    ConvSpeakerEncoder,
    Disentangler,
    LogMelFrontend,
    TrialResult,
    apply_environment,
    build_manifest,
    compute_eer,
    compute_min_dcf,
    crop_or_pad,
    ensure_dir,
    load_audio,
    load_config,
    make_verification_trials,
    plot_score_distributions,
    plot_tsne,
    split_manifest,
)


@torch.no_grad()
def embed_file(path, env_id, frontend, encoder, disentangler, sample_rate, segment_seconds, device):
    waveform = load_audio(path, sample_rate)
    waveform = crop_or_pad(waveform, int(sample_rate * segment_seconds), random_crop=False)
    waveform = apply_environment(waveform, env_id)
    feats = frontend(waveform.to(device)).squeeze(0)
    emb = encoder(feats.unsqueeze(0))
    if disentangler is None:
        return emb.squeeze(0).cpu()
    spk_code, _, _ = disentangler.encode(emb)
    return F.normalize(spk_code, dim=-1).squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    config = load_config(config_path)
    q2_dir = Path(__file__).resolve().parent
    run_name = checkpoint_path.parent.name

    dataset_root = q2_dir / config["dataset"]["root"]
    manifest_path = q2_dir / "results" / "tables" / "manifest.json"
    split_path = q2_dir / "results" / "tables" / "splits.json"
    rows = build_manifest(dataset_root, manifest_path)
    splits = split_manifest(rows, split_path, config["train_ratio"], config["val_ratio"], config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    frontend = LogMelFrontend(config["sample_rate"], config["n_mels"]).to(device)
    encoder = ConvSpeakerEncoder(config["n_mels"], config["embedding_dim"]).to(device)
    frontend.load_state_dict(ckpt["frontend"])
    encoder.load_state_dict(ckpt["encoder"])
    frontend.eval()
    encoder.eval()

    disentangler = None
    if ckpt["model_type"] != "baseline":
        disentangler = Disentangler(
            embedding_dim=config["embedding_dim"],
            latent_dim=config["latent_dim"],
            n_speakers=40,
            n_envs=len(config["env_names"]),
        ).to(device)
        disentangler.load_state_dict(ckpt["disentangler"])
        disentangler.eval()

    trials = make_verification_trials(
        splits["test"],
        num_target=config["num_eval_target_trials"],
        num_impostor=config["num_eval_impostor_trials"],
        seed=config["seed"],
    )

    matched_results = []
    mismatched_results = []
    tsne_embeddings = []
    tsne_speakers = []

    for idx, (left, right, label) in enumerate(trials):
        env_same = idx % len(ENVIRONMENTS)
        env_diff = (env_same + 1) % len(ENVIRONMENTS)

        left_same = embed_file(left["path"], env_same, frontend, encoder, disentangler, config["sample_rate"], config["segment_seconds"], device)
        right_same = embed_file(right["path"], env_same, frontend, encoder, disentangler, config["sample_rate"], config["segment_seconds"], device)
        left_diff = embed_file(left["path"], env_same, frontend, encoder, disentangler, config["sample_rate"], config["segment_seconds"], device)
        right_diff = embed_file(right["path"], env_diff, frontend, encoder, disentangler, config["sample_rate"], config["segment_seconds"], device)

        matched_results.append(TrialResult(F.cosine_similarity(left_same, right_same, dim=0).item(), label, "matched"))
        mismatched_results.append(TrialResult(F.cosine_similarity(left_diff, right_diff, dim=0).item(), label, "mismatched"))

        if idx < 60:
            tsne_embeddings.append(left_diff.numpy())
            tsne_speakers.append(left["speaker_id"])

    summary_rows = []
    for condition, results in (("matched", matched_results), ("mismatched", mismatched_results)):
        labels = np.array([item.label for item in results])
        scores = np.array([item.score for item in results])
        eer, threshold = compute_eer(labels, scores)
        summary_rows.append(
            {
                "run_name": run_name,
                "condition": condition,
                "eer": round(eer * 100, 3),
                "min_dcf": round(compute_min_dcf(labels, scores), 5),
                "threshold": round(threshold, 5),
            }
        )

    tables_dir = q2_dir / "results" / "tables"
    plots_dir = q2_dir / "results" / "plots"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    summary_path = tables_dir / f"{run_name}_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_score_distributions(matched_results, plots_dir / f"{run_name}_matched_scores.png", f"{run_name} matched scores")
    plot_score_distributions(mismatched_results, plots_dir / f"{run_name}_mismatched_scores.png", f"{run_name} mismatched scores")
    plot_tsne(np.array(tsne_embeddings), tsne_speakers, plots_dir / f"{run_name}_tsne.png", f"{run_name} t-SNE")
    print(f"saved metrics to {summary_path}")


if __name__ == "__main__":
    main()
