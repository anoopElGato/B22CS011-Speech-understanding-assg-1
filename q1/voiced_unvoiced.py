import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and preprocess audio
# -----------------------------
sample_rate, signal = wav.read("q1\data\MLK_1.wav")
signal = signal.astype(float)

# Convert stereo → mono
if signal.ndim == 2:
    signal = np.mean(signal, axis=1)

# Normalize
signal = signal / np.max(np.abs(signal))

# -----------------------------
# 2. Framing
# -----------------------------
def framing(signal, sr, frame_size=0.025, frame_stride=0.01):
    frame_length = int(frame_size * sr)
    frame_step = int(frame_stride * sr)

    signal_length = len(signal)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

    pad_length = num_frames * frame_step + frame_length
    pad_signal = np.append(signal, np.zeros(pad_length - signal_length))

    indices = (
        np.tile(np.arange(frame_length), (num_frames, 1)) +
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )

    return pad_signal[indices.astype(np.int32)]

frames = framing(signal, sample_rate)

# -----------------------------
# 3. Windowing (Hamming)
# -----------------------------
frames *= np.hamming(frames.shape[1])

# -----------------------------
# 4. Cepstrum computation
# -----------------------------
def compute_cepstrum(frames, n_fft=512):
    spectrum = np.fft.fft(frames, n=n_fft)
    log_mag = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_mag).real
    return cepstrum

cepstra = compute_cepstrum(frames)

# -----------------------------
# 5. Quefrency split
# -----------------------------
def quefrency_split(cepstra, sr, pitch_range=(50, 400)):
    """
    pitch_range in Hz → convert to quefrency (seconds)
    """
    q_min = int(sr / pitch_range[1])  # high pitch → low quefrency
    q_max = int(sr / pitch_range[0])  # low pitch → high quefrency

    low_energy = []
    high_energy = []

    for c in cepstra:
        # Low quefrency (vocal tract)
        low = np.sum(c[:q_min] ** 2)

        # High quefrency (pitch region)
        high = np.sum(c[q_min:q_max] ** 2)

        low_energy.append(low)
        high_energy.append(high)

    return np.array(low_energy), np.array(high_energy)

low_energy, high_energy = quefrency_split(cepstra, sample_rate)

# -----------------------------
# 6. Voiced / Unvoiced decision
# -----------------------------
ratio = high_energy / (low_energy + 1e-10)

# Threshold (can tune)
threshold = np.median(ratio)

voiced_flags = ratio > threshold  # 1 = voiced, 0 = unvoiced

# -----------------------------
# 7. Time axis
# -----------------------------
frame_stride = 0.01
time_axis = np.arange(len(voiced_flags)) * frame_stride

# -----------------------------
# 8. Visualization
# -----------------------------
plt.figure(figsize=(12, 10))

# Waveform
plt.subplot(4, 1, 1)
t = np.arange(len(signal)) / sample_rate
plt.plot(t, signal)
plt.title("Waveform")

# Cepstral energies
plt.subplot(4, 1, 2)
plt.plot(time_axis, low_energy, label="Low Quefrency (Envelope)")
plt.plot(time_axis, high_energy, label="High Quefrency (Pitch)")
plt.legend()
plt.title("Cepstral Energy")

# Ratio
plt.subplot(4, 1, 3)
plt.plot(time_axis, ratio)
plt.axhline(threshold, linestyle='--')
plt.title("High/Low Energy Ratio")

# Voiced / Unvoiced
plt.subplot(4, 1, 4)
plt.plot(time_axis, voiced_flags.astype(int))
plt.title("Voiced (1) / Unvoiced (0)")

plt.tight_layout()
plt.show()