# Speech Understanding Assignment 1

This repository includes runnable scripts for Question 1 inside `q1/`.

The four scripts covered here are:

- `mfcc_manual.py`
- `leakage_snr.py`
- `voiced_unvoiced.py`
- `phonetic_mapping.py`

## Project Structure

```text
Assg1/
`-- q1/
    |-- data/
    |   |-- MLK_1.wav
    |   |-- MLK_2.wav
    |   `-- MLK_3.wav
    |-- mfcc_manual.py
    |-- leakage_snr.py
    |-- voiced_unvoiced.py
    |-- README.md
    `-- phonetic_mapping.py
```

## Python Setup

Use Python 3.10 or newer.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install numpy scipy pandas matplotlib librosa torch transformers
```

## How To Run

Run all commands from the repository root:

```powershell
cd "c:\Users\AnoopPC\Desktop\assg\Speech Understanding\Assg1"
```

### 1. Run `mfcc_manual.py`

This script manually computes MFCC features from `q1\data\MLK_1.wav`.

```powershell
python q1\mfcc_manual.py
```

Expected output:

- Prints the MFCC matrix shape
- Uses the built-in hardcoded audio path `q1\data\MLK_1.wav`

### 2. Run `leakage_snr.py`

This script compares spectral leakage and SNR for rectangular, Hamming, and Hanning windows.

```powershell
python q1\leakage_snr.py
```

Expected output:

- Prints a table with spectral leakage and SNR values
- Opens matplotlib bar plots

Note:

- The current script uses a synthetic sinusoidal signal, not a `.wav` file
- There is commented code inside the file if you want to switch to `q1\data\MLK_1.wav`

### 3. Run `voiced_unvoiced.py`

This script performs voiced/unvoiced classification using cepstral analysis on `q1\data\MLK_1.wav`.

```powershell
python q1\voiced_unvoiced.py
```

Expected output:

- Opens plots for waveform, cepstral energy, energy ratio, and voiced/unvoiced decisions

### 4. Run `phonetic_mapping.py`

This script loads a pretrained Wav2Vec2 model, decodes speech from `q1\data\MLK_2.wav`, and compares model boundaries with manually defined segment boundaries.

```powershell
python q1\phonetic_mapping.py
```

Expected output:

- Prints decoded text
- Prints a few predicted segments
- Prints boundary RMSE in seconds
- Opens a boundary comparison plot

Important:

- The first run may take time because `transformers` downloads `facebook/wav2vec2-base-960h`
- Internet access is required for that first model download unless it is already cached locally

## Notes

- These scripts currently use hardcoded file paths, so they should be run from the repository root
- If matplotlib windows do not appear, make sure your Python environment supports GUI plotting
- `phonetic_mapping.py` expects 16 kHz audio and already resamples using `librosa.load(..., sr=16000)`

## Quick Run Summary

```powershell
python q1\mfcc_manual.py
python q1\leakage_snr.py
python q1\voiced_unvoiced.py
python q1\phonetic_mapping.py
```
