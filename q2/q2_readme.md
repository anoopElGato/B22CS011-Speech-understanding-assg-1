# Q2 Reproduction Notes

This folder contains a justified reduced reproduction of "Disentangled Representation Learning for Environment-agnostic Speaker Recognition" rather than a literal VoxCeleb2 reproduction. The original paper depends on VoxCeleb2 session metadata, large speaker backbones, and benchmark sets such as VoxSRC22/23 and VC-Mix. Here I preserve the key ideas on a smaller reproducible setup built from LibriSpeech `dev-clean` with synthetic environment labels.

## What Was Implemented

- `train.py`: trains one of three configurations.
- `eval.py`: evaluates a checkpoint as a speaker verification system under matched and mismatched synthetic environments.
- `configs/baseline.json`: compact speaker encoder baseline.
- `configs/proposed.json`: reduced reproduction of the paper's disentangler.
- `configs/improved.json`: critique-motivated variant with explicit environment-label supervision on the environment code.

## Dataset

The experiments use LibriSpeech `dev-clean`, copied into `q2/dataset/dev-clean/`. A manifest and fixed split are saved under `q2/results/tables/`.

Split sizes used in the runs:

- Train: see `q2/results/checkpoints/*/split_sizes.json`
- Validation: see `q2/results/checkpoints/*/split_sizes.json`
- Test: see `q2/results/checkpoints/*/split_sizes.json`

## Reproducing The Runs

From the repository root:

```bash
python q2/train.py --config q2/configs/baseline.json
python q2/eval.py --config q2/configs/baseline.json --checkpoint q2/results/checkpoints/baseline/best.pt

python q2/train.py --config q2/configs/proposed.json
python q2/eval.py --config q2/configs/proposed.json --checkpoint q2/results/checkpoints/proposed/best.pt

python q2/train.py --config q2/configs/improved.json
python q2/eval.py --config q2/configs/improved.json --checkpoint q2/results/checkpoints/improved/best.pt
```

## Checkpoints Corresponding To Reported Results

- Baseline results come from `best.pt` in `results/checkpoints/baseline/best.pt`
- Proposed reduced reproduction results come from `best.pt` in `results/checkpoints/proposed/best.pt`
- Improved model results come from `best.pt` in `results/checkpoints/improved/best.pt`

## Reported Metrics

Summary table: `q2/results/tables/summary_metrics.csv`

Key results:

- Baseline: matched EER 26.0, mismatched EER 28.0
- Proposed: matched EER 37.0, mismatched EER 39.25
- Improved: matched EER 36.0, mismatched EER 35.0

Interpretation:

- In this reduced setting, the paper-style disentangler underperformed the baseline.
- The proposed improvement helped relative to the reduced reproduction on the mismatched condition, but still did not surpass the baseline.
- This is discussed in detail in `review.pdf`.
