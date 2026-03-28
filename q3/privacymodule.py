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