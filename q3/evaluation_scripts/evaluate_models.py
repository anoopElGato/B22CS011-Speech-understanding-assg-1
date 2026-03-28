import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from jiwer import wer
import torchaudio
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from text_processing import normalize_transcript, is_valid_english_transcript

# Transformations
import librosa
import numpy as np
import random
import torch

def pitch_shift(waveform, sr=16000):
    shift = random.uniform(-4, 4)
    return torch.tensor(librosa.effects.pitch_shift(waveform.numpy(), sr=sr, n_steps=shift))

def time_stretch(waveform):
    rate = random.uniform(0.85, 1.15)
    stretched = librosa.effects.time_stretch(waveform.numpy(), rate=rate)
    return torch.tensor(stretched)

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
DATA_ROOT = os.path.join("..", "mozilla_common_voice")
TSV_PATH = os.path.join(DATA_ROOT, "ss-corpus-en.tsv")

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

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

test_df = load_split("test")
test_dataset = SpeechDataset(test_df, processor, apply_privacy=False)

def evaluate_model(model, dataset):
    model.eval()

    results = []
    group_errors = {}

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        input_values = sample["input_values"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)

        pred_text = processor.batch_decode(pred_ids)[0]
        true_text = sample["transcription"]
        
        error = wer(true_text, pred_text)
        results.append(error)

        group = sample["gender"]

        if group not in group_errors:
            group_errors[group] = []

        group_errors[group].append(error)

    overall = np.mean(results)
    group_avg = {k: np.mean(v) for k, v in group_errors.items()}
    gap = max(group_avg.values()) - min(group_avg.values())

    return overall, group_avg, gap

def load_model(path):
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_no_fair = load_model("../model_no_fairness_5.pt")
model_fair = load_model("../model_with_fairness_5.pt")

print("\nEvaluating the model trained WITHOUT fairness loss...")
wer_no_fair, group_no_fair, gap_no_fair = evaluate_model(model_no_fair, test_dataset)
print("\nEvaluating the model trained WITH fairness loss...")
wer_fair, group_fair, gap_fair = evaluate_model(model_fair, test_dataset)

print("\n--- RESULTS ---")
print("No Fairness WER:", wer_no_fair)
print("With Fairness WER:", wer_fair)

print("\nGroup-wise WER (No Fairness):", group_no_fair)
print("Group-wise WER (Fairness):", group_fair)

print("\nFairness Gap (No Fairness):", gap_no_fair)
print("Fairness Gap (Fairness):", gap_fair)

plt.figure()
plt.bar(["No Fairness", "With Fairness"], [wer_no_fair, wer_fair])
plt.title("Overall WER Comparison")
plt.ylabel("WER")
plt.show()

plt.figure()
plt.bar(["No Fairness", "With Fairness"], [gap_no_fair, gap_fair])
plt.title("Fairness Gap Comparison")
plt.ylabel("WER Gap")
plt.show()

groups = list(set(list(group_no_fair.keys()) + list(group_fair.keys())))

no_fair_vals = [group_no_fair.get(g, 0) for g in groups]
fair_vals = [group_fair.get(g, 0) for g in groups]

x = np.arange(len(groups))

plt.figure()
plt.bar(x - 0.2, no_fair_vals, width=0.4, label="No Fairness")
plt.bar(x + 0.2, fair_vals, width=0.4, label="With Fairness")

plt.xticks(x, groups)
plt.title("Group-wise WER Comparison")
plt.legend()
plt.show()
