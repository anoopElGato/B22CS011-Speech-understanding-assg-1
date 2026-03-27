import numpy as np
import scipy.io.wavfile as wav

class MFCCExtractor:
    def __init__(
        self,
        sample_rate=16000,
        frame_size=0.025,
        frame_stride=0.01,
        num_filters=26,
        num_coeffs=13,
        n_fft=512,
        pre_emphasis=0.97,
        window_type="hamming"  # "hamming", "hanning", "rectangular"
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.num_filters = num_filters
        self.num_coeffs = num_coeffs
        self.n_fft = n_fft
        self.pre_emphasis = pre_emphasis
        self.window_type = window_type.lower()

    # -----------------------------
    # 1. Pre-emphasis
    # -----------------------------
    def pre_emphasize(self, signal):
        return np.append(signal[0], signal[1:] - self.pre_emphasis * signal[:-1])

    # -----------------------------
    # 2. Framing
    # -----------------------------
    def framing(self, signal):
        frame_length = int(self.frame_size * self.sample_rate)
        frame_step = int(self.frame_stride * self.sample_rate)

        signal_length = len(signal)
        num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

        pad_length = num_frames * frame_step + frame_length
        pad_signal = np.append(signal, np.zeros(pad_length - signal_length))

        indices = (
            np.tile(np.arange(frame_length), (num_frames, 1)) +
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        )

        frames = pad_signal[indices.astype(np.int32)]
        return frames

    # -----------------------------
    # 3. Windowing
    # -----------------------------
    def apply_window(self, frames):
        frame_length = frames.shape[1]

        if self.window_type == "hamming":
            window = np.hamming(frame_length)
        elif self.window_type == "hanning":
            window = np.hanning(frame_length)
        elif self.window_type == "rectangular":
            window = np.ones(frame_length)
        else:
            raise ValueError(f"Unsupported window type: {self.window_type}")

        return frames * window

    # -----------------------------
    # 4. FFT + Power Spectrum
    # -----------------------------
    def power_spectrum(self, frames):
        fft_frames = np.fft.rfft(frames, self.n_fft)
        power_spec = (1.0 / self.n_fft) * (np.abs(fft_frames) ** 2)
        return power_spec

    # -----------------------------
    # 5. Mel Filterbank
    # -----------------------------
    def hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(self, mel):
        return 700 * (10**(mel / 2595) - 1)

    def mel_filterbank(self):
        low_mel = self.hz_to_mel(0)
        high_mel = self.hz_to_mel(self.sample_rate / 2)

        mel_points = np.linspace(low_mel, high_mel, self.num_filters + 2)
        hz_points = self.mel_to_hz(mel_points)

        bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        fbank = np.zeros((self.num_filters, int(self.n_fft / 2 + 1)))

        for m in range(1, self.num_filters + 1):
            f_m_minus = bins[m - 1]
            f_m = bins[m]
            f_m_plus = bins[m + 1]

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-8)

            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-8)

        return fbank

    # -----------------------------
    # 6. Apply Filterbank + Log
    # -----------------------------
    def apply_filterbank(self, power_spec):
        fbank = self.mel_filterbank()
        filter_banks = np.dot(power_spec, fbank.T)

        # Numerical stability
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)

        log_fbank = np.log(filter_banks)
        return log_fbank

    # -----------------------------
    # 7. DCT (manual implementation)
    # -----------------------------
    def dct(self, x):
        N = x.shape[1]
        result = np.zeros((x.shape[0], self.num_coeffs))

        for k in range(self.num_coeffs):
            result[:, k] = np.sum(
                x * np.cos(np.pi * k * (2 * np.arange(N) + 1) / (2 * N)),
                axis=1
            )

        return result

    # -----------------------------
    # Full Pipeline
    # -----------------------------
    def extract(self, signal):
        emphasized = self.pre_emphasize(signal)
        frames = self.framing(emphasized)
        windowed = self.apply_window(frames)
        power_spec = self.power_spectrum(windowed)
        log_fbank = self.apply_filterbank(power_spec)
        mfcc = self.dct(log_fbank)

        return mfcc
    
if __name__ == '__main__':
    # Load audio
    sample_rate, signal = wav.read("q1\data\MLK_1.wav")

    # Normalize if needed
    signal = signal.astype(float)

    mfcc_extractor = MFCCExtractor(
        sample_rate=sample_rate,
        window_type="hamming"  # try "hanning" or "rectangular"
    )

    mfcc_features = mfcc_extractor.extract(signal)

    print("MFCC shape:", mfcc_features.shape)