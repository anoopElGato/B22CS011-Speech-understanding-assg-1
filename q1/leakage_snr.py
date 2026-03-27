import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load audio
# -----------------------------
# sample_rate, signal = wav.read("q1\data\MLK_1.wav")
# signal = signal.astype(float)

# Synthetic signal
sample_rate = 16000
t = np.linspace(0, 1, sample_rate)
signal = np.sin(2 * np.pi * 440 * t)

# convert to mono audio if stereo (2 channel)
if signal.ndim == 2:
    signal = np.mean(signal, axis=1)

# Use a short speech segment (e.g., first 0.5 sec)
segment = signal[:int(0.5 * sample_rate)]

# -----------------------------
# Window functions
# -----------------------------
def get_window(window_type, N):
    if window_type == "hamming":
        return np.hamming(N)
    elif window_type == "hanning":
        return np.hanning(N)
    elif window_type == "rectangular":
        return np.ones(N)
    else:
        raise ValueError("Invalid window type")

# -----------------------------
# FFT Power Spectrum
# -----------------------------
def compute_spectrum(signal, window):
    windowed = signal * window
    fft_vals = np.fft.rfft(windowed)
    power = np.abs(fft_vals) ** 2
    return power

# -----------------------------
# Spectral Leakage
# -----------------------------

def spectral_leakage(power_spectrum, main_lobe_width=5):
    peak_idx = np.argmax(power_spectrum)

    left = max(0, peak_idx - main_lobe_width)
    right = min(len(power_spectrum), peak_idx + main_lobe_width)

    main_lobe_energy = np.sum(power_spectrum[left:right])
    total_energy = np.sum(power_spectrum)

    leakage = (total_energy - main_lobe_energy) / total_energy
    return leakage

# -----------------------------
# SNR
# -----------------------------

def compute_snr(power_spectrum, main_lobe_width=5):
    peak_idx = np.argmax(power_spectrum)

    left = max(0, peak_idx - main_lobe_width)
    right = min(len(power_spectrum), peak_idx + main_lobe_width)

    signal_power = np.sum(power_spectrum[left:right])
    noise_power = np.sum(power_spectrum) - signal_power

    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr

# -----------------------------
# Run analysis
# -----------------------------
window_types = ["rectangular", "hamming", "hanning"]
results = []

N = len(segment)

for wtype in window_types:
    window = get_window(wtype, N)
    power_spec = compute_spectrum(segment, window)

    leakage = spectral_leakage(power_spec, main_lobe_width= 20)
    # leakage = spectral_leakage(power_spec)
    snr = compute_snr(power_spec, main_lobe_width= 20)
    # snr = compute_snr(power_spec)

    results.append({
        "Window": wtype.capitalize(),
        "Spectral Leakage": leakage,
        "SNR (dB)": snr
    })

df = pd.DataFrame(results)
print(df)

# Spectral Leakage Plot
plt.figure()
plt.bar(df["Window"], df["Spectral Leakage"])
plt.title("Spectral Leakage Comparison")
plt.xlabel("Window Type")
plt.ylabel("Leakage")
plt.show()

# SNR Plot
plt.figure()
plt.bar(df["Window"], df["SNR (dB)"])
plt.title("SNR Comparison")
plt.xlabel("Window Type")
plt.ylabel("SNR (dB)")
plt.show()

plt.figure()