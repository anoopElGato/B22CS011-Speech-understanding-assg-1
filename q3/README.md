# Q3 README

This folder contains the code for Question 3:

1. `audit.py`
2. `privacymodule.py`
3. `pp_demo.py`
4. `train_fair.py`
5. `evaluation_scripts/evaluate_models.py`
6. `evaluation_scripts/evaluate_ppt.py`

## Dataset

The dataset folder `mozilla_common_voice` is **not uploaded to GitHub**.

Download the dataset from:

`https://datacollective.mozillafoundation.org/datasets/cmn1pv5hi00uto1072y1074y7`

After downloading, place it inside `q3/` with this structure:

```text
q3/
|-- mozilla_common_voice/
|   |-- ss-corpus-en.tsv
|   `-- audios/
|       |-- <audio files>
|-- audit.py
|-- privacymodule.py
|-- pp_demo.py
|-- train_fair.py
`-- evaluation_scripts/
```

## Python Setup

Use Python 3.10 or newer.

Install the main dependencies:

```powershell
pip install torch torchaudio transformers pandas matplotlib seaborn librosa soundfile jiwer tqdm numpy scipy
```

Extra package for privacy-quality evaluation:

```powershell
pip install torchmetrics onnxruntime requests
```

Optional package for FAD experiments:

```powershell
pip install frechet-audio-distance
```

Note:

- The Hugging Face model `facebook/wav2vec2-base-960h` is downloaded automatically on first use
- `evaluate_ppt.py` currently computes DNSMOS; the FAD section is commented out in the script

## Where To Run From

Because these scripts use relative paths, run them from the correct working directory:

- Run `audit.py`, `train_fair.py`, and module-level helpers from inside `q3/`
- Run `evaluate_models.py` and `evaluate_ppt.py` from inside `q3/evaluation_scripts/`
- `pp_demo.py` uses `q1\data\MLK_1.wav`, so run it from the repository root

## 1. `audit.py`

Purpose:

- Loads a dataset split from `mozilla_common_voice/ss-corpus-en.tsv`
- Filters invalid or non-English transcripts
- Prints missing-value statistics
- Plots gender and age distributions

Run:

```powershell
cd q3
python audit.py
```

Expected output:

- Console summary for missing values, gender distribution, and age distribution
- Two matplotlib plots

## 2. `privacymodule.py`

Purpose:

- Defines the privacy transforms used in Q3:
  - pitch shifting
  - time stretching
  - additive noise

This file is mainly used a helper module and is typically imported by other scripts instead of being run directly.

Example usage:

```python
from privacymodule import apply_privacy_transforms
```

## 3. `pp_demo.py`

Purpose:

- Demonstrates privacy-preserving transformation on a single audio file
- Loads `q1\data\MLK_1.wav`
- Applies the privacy module
- Saves the transformed result as `output_transformed.wav`

Run from the repository root:

```powershell
python q3\pp_demo.py
```

Expected output:

- Creates `output_transformed.wav`

## 4. `train_fair.py`

Purpose:

- Loads the train split from the Mozilla Common Voice dataset
- Applies privacy transforms to training audio
- Fine-tunes two Wav2Vec2 models:
  - with fairness loss
  - without fairness loss

Run:

```powershell
cd q3
python train_fair.py
```

Expected output:

- Training progress for each epoch
- Saved checkpoints such as:
  - `model_with_fairness_1.pt`
  - `model_with_fairness_2.pt`
  - `model_no_fairness_1.pt`
  - `model_no_fairness_2.pt`

Notes:

- The script currently trains for 5 epochs by default
- It expects `mozilla_common_voice/ss-corpus-en.tsv` and `mozilla_common_voice/audios/` inside `q3/`
- Since `mozilla_common_voice` is not uploaded on git, download it from [here](https://datacollective.mozillafoundation.org/datasets/cmn1pv5hi00uto1072y1074y7) before training

## 5. `evaluation_scripts/evaluate_models.py`

Purpose:

- Loads the test split
- Loads the trained checkpoints
- Computes overall WER
- Computes group-wise WER
- Computes the fairness gap
- Plots comparison charts

Run:

```powershell
cd q3\evaluation_scripts
python evaluate_models.py
```

Expected output:

- Overall WER for both models
- Group-wise WER by gender
- Fairness gap
- Matplotlib plots for WER and fairness comparison

Important:

- This script expects these checkpoint files in `q3/`:
  - `model_no_fairness_5.pt`
  - `model_with_fairness_5.pt`
- If your filenames differ, update the paths inside the script

## 6. `evaluation_scripts/evaluate_ppt.py`

Purpose:

- Evaluates the effect of privacy transformations on audio quality
- Saves original and transformed test audio samples
- Computes DNSMOS-based quality scores
- Plots DNSMOS distributions and comparisons

Run:

```powershell
cd q3\evaluation_scripts
python evaluate_ppt.py
```

Expected output:

- Saved audio files inside:
  - `original_audio/`
  - `transformed_audio/`
- Average DNSMOS scores for original vs transformed audio
- DNSMOS plots

Notes:

- DNSMOS runs on CPU in the current script
- If no valid DNSMOS scores are produced, the plotting section is skipped
- The FAD block is currently commented out

## Quick Run Summary

```powershell
cd q3
python audit.py
python train_fair.py

cd evaluation_scripts
python evaluate_models.py
python evaluate_ppt.py

cd ..\..
python q3\pp_demo.py
```
