import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import Wav2Vec2Processor

from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
# from frechet_audio_distance import FrechetAudioDistance

import soundfile as sf

# Transformations
import librosa
import numpy as np
import random
import torch

def pitch_shift(waveform, sr=16000):
    shift = random.uniform(-4, 4)
    shifted = librosa.effects.pitch_shift(waveform.squeeze(0).numpy(), sr=sr, n_steps=shift)
    return torch.tensor(shifted, dtype=waveform.dtype).unsqueeze(0)

def time_stretch(waveform):
    rate = random.uniform(0.85, 1.15)
    stretched = librosa.effects.time_stretch(waveform.squeeze(0).numpy(), rate=rate)
    return torch.tensor(stretched, dtype=waveform.dtype).unsqueeze(0)

def add_noise(waveform):
    noise = torch.randn_like(waveform) * 0.005
    return waveform + noise

def apply_privacy_transforms(waveform):
    if random.random() < 0.5:
        waveform = pitch_shift(waveform)
    if random.random() < 0.5:
        waveform = time_stretch(waveform)
    if random.random() < 0.3:
        waveform = add_noise(waveform)
    return waveform

# load split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
DATA_ROOT = os.path.join("..", "mozilla_common_voice")
TSV_PATH = os.path.join(DATA_ROOT, "ss-corpus-en.tsv")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from text_processing import normalize_transcript, is_valid_english_transcript

def load_split(split="train"):
    df = pd.read_csv(TSV_PATH, sep="\t")

    # -------------------------------
    # Filter by split
    # -------------------------------
    df = df[df["split"] == split]

    # -------------------------------
    # Basic cleanup
    # -------------------------------
    df = df.dropna(subset=["audio_file", "transcription"])
    df = df[df["audio_file"].apply(lambda x: isinstance(x, str))]
    df = df[df["transcription"].apply(is_valid_english_transcript)]

    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} samples for split: {split}")
    return df

# dataset
CLIPS_PATH = os.path.join(DATA_ROOT, "audios")

device = "cuda" if torch.cuda.is_available() else "cpu"

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, apply_privacy=True):
        df = df[df["transcription"].apply(is_valid_english_transcript)].copy()
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.apply_privacy = apply_privacy

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_path = os.path.join(CLIPS_PATH, row["audio_file"])

        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # Apply privacy transformation
        if self.apply_privacy:
            waveform = apply_privacy_transforms(waveform)

        input_values = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values[0]

        transcript = normalize_transcript(row["transcription"])
        labels = self.processor(text=transcript).input_ids

        return {
            "input_values": input_values,
            "labels": torch.tensor(labels),
            "transcription": transcript,
            "gender": row["gender"],
            "age": row["age"],
            "audio_path": audio_path
        }

test_df = load_split("test")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
test_dataset = SpeechDataset(test_df, processor, apply_privacy=False)

os.makedirs("original_audio", exist_ok=True)
os.makedirs("transformed_audio", exist_ok=True)

DNSMOS_SCORE_NAMES = ["p808_mos", "mos_sig", "mos_bak", "mos_ovr"]


def compute_dnsmos_scores(waveform, sr=16000):
    scores = deep_noise_suppression_mean_opinion_score(
        waveform.squeeze(0).cpu(),
        sr,
        personalized=False,
        device="cpu",
    )
    return {
        name: float(value)
        for name, value in zip(DNSMOS_SCORE_NAMES, scores.tolist())
    }

def evaluate_privacy_effect(dataset, num_samples=200):
    dnsmos_original = []
    dnsmos_transformed = []

    for i, sample in enumerate(tqdm(dataset)):
        if i >= num_samples:
            break

        try:
            waveform, sr = torchaudio.load(sample["audio_path"])
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

            # Save original
            orig_path = f"original_audio/{i}.wav"
            sf.write(orig_path, waveform.squeeze().numpy(), 16000)

            # Apply transformation
            transformed = apply_privacy_transforms(waveform)

            # Save transformed
            trans_path = f"transformed_audio/{i}.wav"
            sf.write(trans_path, transformed.squeeze().numpy(), 16000)

            # -----------------------
            # DNSMOS
            # -----------------------
            mos_orig = compute_dnsmos_scores(waveform, 16000)
            mos_trans = compute_dnsmos_scores(transformed, 16000)

            dnsmos_original.append(mos_orig["mos_ovr"])
            dnsmos_transformed.append(mos_trans["mos_ovr"])

        except Exception as e:
            print(e)
            continue

    return dnsmos_original, dnsmos_transformed

dnsmos_orig, dnsmos_trans = evaluate_privacy_effect(test_dataset, num_samples=200)

if dnsmos_orig and dnsmos_trans:
    print("DNSMOS Original:", np.mean(dnsmos_orig))
    print("DNSMOS Transformed:", np.mean(dnsmos_trans))
else:
    print("DNSMOS could not be computed for any sample.")

# fad_score = None
# try:
#     fad = FrechetAudioDistance()
#     fad_score = fad.score("original_audio", "transformed_audio")
#     print("FAD Score:", fad_score)
# except Exception as e:
#     print(f"Skipping FAD computation: {e}")

if dnsmos_orig and dnsmos_trans:
    plt.figure()
    plt.hist(dnsmos_orig, bins=20, alpha=0.5, label="Original")
    plt.hist(dnsmos_trans, bins=20, alpha=0.5, label="Transformed")
    plt.legend()
    plt.title("DNSMOS Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure()
    plt.bar(
        ["Original", "Transformed"],
        [np.mean(dnsmos_orig), np.mean(dnsmos_trans)]
    )
    plt.title("Average DNSMOS")
    plt.ylabel("Score")
    plt.show()

    plt.figure()
    plt.boxplot([dnsmos_orig, dnsmos_trans], labels=["Original", "Transformed"])
    plt.title("DNSMOS Comparison")
    plt.ylabel("Score")
    plt.show()
