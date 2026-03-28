import librosa
import torch
import soundfile as sf
from privacymodule import apply_privacy_transforms

def transform_audio_file(input_path, output_path):
    waveform, sr = librosa.load(input_path, sr=16000)
    waveform = torch.tensor(waveform)

    transformed = apply_privacy_transforms(waveform)

    sf.write(output_path, transformed.numpy(), 16000)

# Example
transform_audio_file("q1\data\MLK_1.wav", "output_transformed.wav")