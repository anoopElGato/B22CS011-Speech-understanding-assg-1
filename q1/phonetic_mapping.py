# 1. Load Model + Audio

import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio (16kHz required)
audio, sr = librosa.load("q1\data\MLK_1.wav", sr=16000)

input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

# 2. Get Frame-Level Predictions

with torch.no_grad():
    logits = model(input_values).logits

pred_ids = torch.argmax(logits, dim=-1)
tokens = processor.batch_decode(pred_ids)[0]

print("Decoded text:", tokens)

# 3. Get Time Alignment (Frame -> Time)

frame_duration = 0.02  # approx 20 ms per frame

pred_ids = pred_ids[0].cpu().numpy()

# 4. Collapse CTC Repetitions -> Segments

def extract_segments(pred_ids, frame_duration, processor):
    segments = []
    prev_id = None
    start = 0

    for i, pid in enumerate(pred_ids):
        if pid != prev_id:
            if prev_id is not None and prev_id != processor.tokenizer.pad_token_id:
                segments.append({
                    "label": processor.tokenizer.decode([prev_id]),
                    "start": start * frame_duration,
                    "end": i * frame_duration
                })
            start = i
            prev_id = pid

    return segments

segments = extract_segments(pred_ids, frame_duration, processor)

for s in segments[:10]:
    print(s)

# Manual Segments (from cepstrum)

manual_segments = [
    {"start": 0.28, "end": 0.68},
    {"start": 1.15, "end": 1.56},
    {"start": 2.00, "end": 2.36},
    {"start": 3.23, "end": 3.67},
    {"start": 7.43, "end": 7.69},
    {"start": 10.56, "end": 10.76},
    {"start": 14.40, "end": 14.64},
    {"start": 14.84, "end": 15.14},
]

# 6. Boundary Extraction

def get_boundaries(segments):
    return np.array([s["start"] for s in segments] + [segments[-1]["end"]])

model_boundaries = get_boundaries(segments)
manual_boundaries = get_boundaries(manual_segments)

# Lengths may differ -> need matching
# Simple Matching (Nearest Neighbor)

def match_boundaries(manual, model):
    matched = []
    for m in manual:
        closest = model[np.argmin(np.abs(model - m))]
        matched.append(closest)
    return np.array(matched)

matched_model = match_boundaries(manual_boundaries, model_boundaries)

# 7. RMSE Calculation

def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

rmse = compute_rmse(manual_boundaries, matched_model)

print("Boundary RMSE (seconds):", rmse)

# 8. Visualization

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.plot(manual_boundaries, np.zeros_like(manual_boundaries), 'o', label="Manual")
plt.plot(model_boundaries, np.ones_like(model_boundaries), 'x', label="Model")

plt.yticks([0, 1], ["Manual", "Model"])
plt.title("Boundary Alignment Comparison")
plt.legend()

plt.show()
