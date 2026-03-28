import torch
import torchaudio
import random
from privacymodule import apply_privacy_transforms
import os
from audit import load_split
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from jiwer import wer
import numpy as np
from text_processing import normalize_transcript, is_valid_english_transcript

DATA_ROOT = "mozilla_common_voice"
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
        if not transcript:
            raise ValueError(f"Empty normalized transcript for row {idx}: {row['transcription']!r}")
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
model_with_fair = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model_without_fair = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    # -----------------------
    # Pad audio inputs
    # -----------------------
    batch_inputs = processor.pad(
        {"input_values": input_values},
        return_tensors="pt",
        padding=True
    )

    # Manually create attention mask if missing
    if "attention_mask" not in batch_inputs:
        attention_mask = torch.ones_like(batch_inputs["input_values"])
    else:
        attention_mask = batch_inputs["attention_mask"]

    # -----------------------
    # Pad labels
    # -----------------------
    batch_labels = processor.tokenizer.pad(
        {"input_ids": labels},
        return_tensors="pt",
        padding=True
    )

    labels = batch_labels["input_ids"]

    # Replace padding with -100 for CTC
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "input_values": batch_inputs["input_values"],
        "attention_mask": attention_mask,
        "labels": labels,
        "transcription": [item["transcription"] for item in batch],
        "gender": [item["gender"] for item in batch],
        "age": [item["age"] for item in batch]
    }

def compute_wer_batch(pred_ids, label_ids, processor):
    pred_str = processor.batch_decode(pred_ids)

    # Replace -100 with pad_token_id
    label_ids = label_ids.clone()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    label_str = processor.batch_decode(label_ids, group_tokens=False)

    wers = []
    for p, l in zip(pred_str, label_str):
        try:
            wers.append(wer(l, p))
        except:
            continue

    return wers

def fairness_loss(outputs, batch, processor, lambda_fair=0.1):
    with torch.no_grad():  # IMPORTANT (WER is non-differentiable)
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)

        wers = compute_wer_batch(pred_ids, batch["labels"], processor)

        if len(wers) == 0:
            return torch.tensor(0.0, device=logits.device)

        # Group WERs
        group_wers = {}
        for i, gender in enumerate(batch["gender"]):
            if i >= len(wers):
                continue

            if gender not in group_wers:
                group_wers[gender] = []

            group_wers[gender].append(wers[i])

        # Compute mean WER per group
        group_means = []
        for g in group_wers:
            if len(group_wers[g]) > 0:
                group_means.append(sum(group_wers[g]) / len(group_wers[g]))

        if len(group_means) <= 1:
            return torch.tensor(0.0, device=logits.device)

        gap = max(group_means) - min(group_means)

    return gap

import torch
import torch.nn as nn

ctc_loss_fn = nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)

def fairness_loss_ctc(outputs, batch, lambda_fair=0.1):
    logits = outputs.logits  # (B, T, V)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    input_lengths = torch.full(
        size=(log_probs.size(0),),
        fill_value=log_probs.size(1),
        dtype=torch.long,
        device=log_probs.device
    )

    labels = batch["labels"].to(log_probs.device)

    # Compute label lengths (ignore -100)
    label_lengths = (labels != -100).sum(dim=1)

    # Replace -100 with blank token (0)
    labels_ctc = labels.clone()
    labels_ctc[labels_ctc == -100] = 0

    # CTC expects shape (T, B, V)
    log_probs = log_probs.transpose(0, 1)

    # -----------------------
    # Per-sample CTC loss
    # -----------------------
    losses = ctc_loss_fn(
        log_probs,
        labels_ctc,
        input_lengths,
        label_lengths
    )  # shape: (B,)

    # -----------------------
    # Group-wise aggregation
    # -----------------------
    group_losses = {}

    for i, gender in enumerate(batch["gender"]):
        if gender not in group_losses:
            group_losses[gender] = []

        group_losses[gender].append(losses[i])

    group_means = []
    for g in group_losses:
        group_means.append(torch.stack(group_losses[g]).mean())

    if len(group_means) <= 1:
        return torch.tensor(0.0, device=logits.device)

    group_means = torch.stack(group_means)

    gap = torch.max(group_means) - torch.min(group_means)

    return lambda_fair * gap

def train(model, dataloader, optimizer, epochs= 5, use_fairness=False):
    model.train()
    model.to(device)

    for e in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch{e+1}/{epochs}")
        epoch_loss = 0
        for batch in pbar:
            input_values = batch["input_values"].to(device)
            attn_masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_values=input_values,
                attention_mask=attn_masks,
                labels=labels
            )
            loss = outputs.loss

            if use_fairness:
                f_loss = fairness_loss_ctc(outputs, batch)
                loss = loss + f_loss

            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            else:
                print("Skipping batch due to NaN or Inf loss")
            
        print(f"Avg loss: {epoch_loss/len(dataloader)}\n")
        if (use_fairness):
            save_model(model_with_fair, f"model_with_fairness_{e+1}.pt")
        else:
            save_model(model_without_fair, f"model_no_fairness_{e+1}.pt")

def save_model(model, path):
    torch.save(model.state_dict(), path)

train_df = load_split("train")
train_dataset = SpeechDataset(train_df, processor, apply_privacy=True)

dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

optimizer_fair = torch.optim.AdamW(model_with_fair.parameters(), lr=1e-5)
optimizer_no_fair = torch.optim.AdamW(model_without_fair.parameters(), lr=1e-5)

# Train WITH fairness
print("\nTraining model WITH fairness loss")
train(model_with_fair, dataloader, optimizer_fair, use_fairness=True)

# Train WITHOUT fairness
print("\nTraining model WITHOUT fairness loss")
train(model_without_fair, dataloader, optimizer_no_fair, use_fairness=False)
