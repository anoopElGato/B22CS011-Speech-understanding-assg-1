import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from torch.autograd import Function
from torch.utils.data import Dataset


ENVIRONMENTS = ["clean", "gaussian", "reverb", "bandstop"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_manifest(dataset_root: Path, manifest_path: Path) -> List[Dict]:
    ensure_dir(manifest_path.parent)
    if manifest_path.exists():
        return load_json(manifest_path)

    rows = []
    for audio_path in sorted(dataset_root.rglob("*.flac")):
        rows.append(
            {
                "path": str(audio_path),
                "speaker_id": int(audio_path.parts[-3]),
                "chapter_id": int(audio_path.parts[-2]),
                "utterance_id": audio_path.stem,
            }
        )
    save_json(rows, manifest_path)
    return rows


def split_manifest(
    rows: Sequence[Dict],
    split_path: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Dict]]:
    ensure_dir(split_path.parent)
    if split_path.exists():
        return load_json(split_path)

    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["speaker_id"]].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for speaker_rows in grouped.values():
        chapter_groups: Dict[int, List[Dict]] = defaultdict(list)
        for row in speaker_rows:
            chapter_groups[row["chapter_id"]].append(row)

        ordered = []
        for chapter_id in sorted(chapter_groups):
            chunk = chapter_groups[chapter_id][:]
            rng.shuffle(chunk)
            ordered.extend(chunk)

        total = len(ordered)
        n_train = max(3, int(total * train_ratio))
        n_val = max(1, int(total * val_ratio))
        n_test = max(1, total - n_train - n_val)

        while n_train + n_val + n_test > total:
            if n_train > 3:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_test -= 1

        while n_train + n_val + n_test < total:
            n_train += 1

        splits["train"].extend(ordered[:n_train])
        splits["val"].extend(ordered[n_train : n_train + n_val])
        splits["test"].extend(ordered[n_train + n_val : n_train + n_val + n_test])

    save_json(splits, split_path)
    return splits


def create_label_maps(rows: Sequence[Dict]) -> Tuple[Dict[int, int], Dict[int, int]]:
    speakers = sorted({row["speaker_id"] for row in rows})
    chapters = sorted({row["chapter_id"] for row in rows})
    return (
        {speaker_id: idx for idx, speaker_id in enumerate(speakers)},
        {chapter_id: idx for idx, chapter_id in enumerate(chapters)},
    )


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform


def crop_or_pad(waveform: torch.Tensor, length: int, random_crop: bool) -> torch.Tensor:
    current = waveform.shape[-1]
    if current > length:
        start = random.randint(0, current - length) if random_crop else max(0, (current - length) // 2)
        return waveform[:, start : start + length]
    if current < length:
        return F.pad(waveform, (0, length - current))
    return waveform


def apply_environment(waveform: torch.Tensor, env_id: int) -> torch.Tensor:
    if env_id == 0:
        return waveform
    if env_id == 1:
        noise = torch.randn_like(waveform)
        snr_db = random.uniform(8.0, 18.0)
        signal_power = waveform.pow(2).mean().clamp_min(1e-6)
        noise_power = noise.pow(2).mean().clamp_min(1e-6)
        scale = torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
        return waveform + scale * noise
    if env_id == 2:
        reverb_len = random.randint(800, 2400)
        t = torch.arange(reverb_len, dtype=waveform.dtype)
        rir = torch.exp(-t / random.uniform(80.0, 220.0))
        rir = rir * torch.rand(reverb_len, dtype=waveform.dtype)
        rir[0] = 1.0
        rir = rir / rir.abs().sum().clamp_min(1e-6)
        padded = F.pad(waveform, (0, reverb_len - 1))
        return F.conv1d(padded.unsqueeze(0), rir.view(1, 1, -1)).squeeze(0)[:, : waveform.shape[-1]]
    if env_id == 3:
        spec = torch.fft.rfft(waveform)
        n_bins = spec.shape[-1]
        lo = random.randint(max(1, n_bins // 12), max(2, n_bins // 4))
        hi = random.randint(max(lo + 1, n_bins // 2), n_bins - 1)
        mask = torch.ones(n_bins, dtype=spec.dtype, device=spec.device)
        mask[lo:hi] = 0.35
        return torch.fft.irfft(spec * mask, n=waveform.shape[-1])
    raise ValueError(f"Unknown environment id: {env_id}")


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate: int, n_mels: int) -> None:
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(sample_rate * 0.025),
            win_length=int(sample_rate * 0.025),
            hop_length=int(sample_rate * 0.010),
            n_mels=n_mels,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        feats = self.melspec(waveform).clamp_min(1e-5).log()
        return (feats - feats.mean(dim=-1, keepdim=True)) / (feats.std(dim=-1, keepdim=True) + 1e-5)


class ConvSpeakerEncoder(nn.Module):
    def __init__(self, n_mels: int, embedding_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.features(feats.unsqueeze(1))), dim=-1)


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_)


class Disentangler(nn.Module):
    def __init__(self, embedding_dim: int, latent_dim: int, n_speakers: int, n_envs: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, embedding_dim),
        )
        self.spk_dim = latent_dim // 2
        self.env_dim = latent_dim - self.spk_dim
        self.spk_classifier = nn.Linear(self.spk_dim, n_speakers)
        self.env_head = nn.Sequential(nn.Linear(self.env_dim, self.env_dim), nn.ReLU(), nn.Linear(self.env_dim, 64))
        self.adv_env_head = nn.Sequential(nn.Linear(self.spk_dim, self.spk_dim), nn.ReLU(), nn.Linear(self.spk_dim, 64))
        self.env_classifier = nn.Linear(self.env_dim, n_envs)

    def encode(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code = self.encoder(embeddings)
        spk_code = F.normalize(code[:, : self.spk_dim], dim=-1)
        env_code = F.normalize(code[:, self.spk_dim :], dim=-1)
        recon = self.decoder(torch.cat([spk_code, env_code], dim=-1))
        return spk_code, env_code, recon


def mapc_loss(spk_code: torch.Tensor, env_code: torch.Tensor) -> torch.Tensor:
    spk_centered = spk_code - spk_code.mean(dim=0, keepdim=True)
    env_centered = env_code - env_code.mean(dim=0, keepdim=True)
    cov = (spk_centered * env_centered).mean(dim=0).abs()
    denom = spk_centered.std(dim=0).clamp_min(1e-6) * env_centered.std(dim=0).clamp_min(1e-6)
    return (cov / denom).mean()


def triplet_distance_loss(anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, margin: float) -> torch.Tensor:
    pos_dist = (anchors - positives).pow(2).sum(dim=-1)
    neg_dist = (anchors - negatives).pow(2).sum(dim=-1)
    return F.relu(margin + pos_dist - neg_dist).mean()


@dataclass
class TrialResult:
    score: float
    label: int
    condition: str


class BaselineDataset(Dataset):
    def __init__(self, rows: Sequence[Dict], speaker_to_idx: Dict[int, int], sample_rate: int, segment_seconds: float, training: bool) -> None:
        self.rows = list(rows)
        self.speaker_to_idx = speaker_to_idx
        self.sample_rate = sample_rate
        self.segment_length = int(sample_rate * segment_seconds)
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        env_id = random.randrange(len(ENVIRONMENTS)) if self.training else 0
        waveform = crop_or_pad(load_audio(row["path"], self.sample_rate), self.segment_length, self.training)
        waveform = apply_environment(waveform, env_id)
        return {
            "waveform": waveform,
            "speaker": torch.tensor(self.speaker_to_idx[row["speaker_id"]], dtype=torch.long),
            "env_id": torch.tensor(env_id, dtype=torch.long),
        }


class TripletEnvironmentDataset(Dataset):
    def __init__(self, rows: Sequence[Dict], speaker_to_idx: Dict[int, int], sample_rate: int, segment_seconds: float, steps_per_epoch: int) -> None:
        self.rows = list(rows)
        self.speaker_to_idx = speaker_to_idx
        self.sample_rate = sample_rate
        self.segment_length = int(sample_rate * segment_seconds)
        self.steps_per_epoch = steps_per_epoch
        self.by_speaker_chapter: Dict[int, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        for row in self.rows:
            self.by_speaker_chapter[row["speaker_id"]][row["chapter_id"]].append(row)
        self.speakers = sorted(self.by_speaker_chapter.keys())

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        speaker_id = random.choice(self.speakers)
        chapters = list(self.by_speaker_chapter[speaker_id].keys())
        same_chapter = random.choice(chapters)
        same_rows = self.by_speaker_chapter[speaker_id][same_chapter]
        r1, r2 = random.sample(same_rows, 2) if len(same_rows) >= 2 else (same_rows[0], same_rows[0])
        diff_chapters = [chapter for chapter in chapters if chapter != same_chapter]
        diff_chapter = random.choice(diff_chapters) if diff_chapters else same_chapter
        r3 = random.choice(self.by_speaker_chapter[speaker_id][diff_chapter])

        env_same = random.randrange(len(ENVIRONMENTS))
        env_diff = random.choice([idx for idx in range(len(ENVIRONMENTS)) if idx != env_same])

        waveforms = []
        for row, env_id in ((r1, env_same), (r2, env_same), (r3, env_diff)):
            waveform = crop_or_pad(load_audio(row["path"], self.sample_rate), self.segment_length, True)
            waveforms.append(apply_environment(waveform, env_id))

        return {
            "waveforms": torch.stack(waveforms, dim=0),
            "speaker": torch.tensor(self.speaker_to_idx[speaker_id], dtype=torch.long),
            "env_ids": torch.tensor([env_same, env_same, env_diff], dtype=torch.long),
        }


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def compute_min_dcf(labels: np.ndarray, scores: np.ndarray, p_target: float = 0.05, c_miss: float = 1.0, c_fa: float = 1.0) -> float:
    positives = labels == 1
    negatives = labels == 0
    best = float("inf")
    for threshold in np.unique(scores):
        predicts_positive = scores >= threshold
        fnr = ((~predicts_positive) & positives).sum() / max(1, positives.sum())
        fpr = (predicts_positive & negatives).sum() / max(1, negatives.sum())
        best = min(best, float(c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)))
    return best


def make_verification_trials(rows: Sequence[Dict], num_target: int, num_impostor: int, seed: int) -> List[Tuple[Dict, Dict, int]]:
    rng = random.Random(seed)
    by_speaker: Dict[int, List[Dict]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)

    target_trials = []
    for speaker_rows in by_speaker.values():
        if len(speaker_rows) < 2:
            continue
        pairs = []
        for idx in range(len(speaker_rows)):
            for jdx in range(idx + 1, len(speaker_rows)):
                pairs.append((speaker_rows[idx], speaker_rows[jdx], 1))
        rng.shuffle(pairs)
        target_trials.extend(pairs[: max(1, num_target // max(1, len(by_speaker)))])

    speaker_ids = list(by_speaker.keys())
    impostor_trials = []
    while len(impostor_trials) < num_impostor:
        s1, s2 = rng.sample(speaker_ids, 2)
        impostor_trials.append((rng.choice(by_speaker[s1]), rng.choice(by_speaker[s2]), 0))

    trials = target_trials[:num_target] + impostor_trials[:num_impostor]
    rng.shuffle(trials)
    return trials


def plot_score_distributions(results: List[TrialResult], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    for label, name in ((1, "target"), (0, "impostor")):
        scores = [item.score for item in results if item.label == label]
        plt.hist(scores, bins=30, alpha=0.5, density=True, label=name)
    plt.title(title)
    plt.xlabel("Cosine score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_tsne(embeddings: np.ndarray, speaker_labels: List[int], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    if len(embeddings) < 5:
        return
    perplexity = min(20, max(3, len(embeddings) // 4))
    coords = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=0).fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    speakers = sorted(set(speaker_labels))[:10]
    for speaker in speakers:
        idxs = [idx for idx, value in enumerate(speaker_labels) if value == speaker]
        plt.scatter(coords[idxs, 0], coords[idxs, 1], label=f"spk {speaker}", s=18)
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
