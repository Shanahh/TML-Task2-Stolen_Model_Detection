# Stolen Model Detection

Low-FPR stolen model detector for the Trustworthy Machine Learning 2026 model stealing task.

The script compares one target CIFAR-100 ResNet-18 model against 360 suspect ResNet-18 models and produces continuous stealing-confidence scores. Higher scores mean the suspect model is more likely to be copied, fine-tuned, distilled, extracted, or otherwise derived from the target model.

## Expected File Structure

```text
.
├── submission.py
├── target_model/
│   ├── weights.safetensors
│   └── train_main_idx.json
├── suspect_models/
│   ├── suspect_000.safetensors
│   ├── suspect_001.safetensors
│   ├── ...
│   └── suspect_359.safetensors
└── dataset/
    └── cifar100/
```

`dataset/cifar100/` can be empty. CIFAR-100 is downloaded automatically by `torchvision`.

## Installation

```bash
pip install torch torchvision pandas numpy safetensors
```

Use the correct PyTorch version for your CUDA setup if running on GPU.

## Reproduce the Submission

Run from the repository root:

```bash
python task_template.py \
  --root . \
  --data-root ./dataset/cifar100 \
  --output submission.csv \
  --device auto \
  --seed 123
```

This creates the final leaderboard submission file:

```text
submission.csv
```

Submit this file to the leaderboard.

## Faster Run

FGSM and Jacobian features are slower. To disable them:

```bash
python task_template.py \
  --root . \
  --data-root ./dataset/cifar100 \
  --output submission.csv \
  --device auto \
  --no-fgsm \
  --no-jacobian
```

## Re-score Existing Features

Feature extraction is expensive. After one full run, reuse the extracted features:

```bash
python task_template.py \
  --features-csv submission_features.csv \
  --output submission_rescored.csv
```

This only recomputes the final scores.

## Produced Files

A normal run with `--output submission.csv` writes:

```text
submission.csv
submission_features.csv
submission_diagnostics.txt
submission_diagnostics.csv
```

### `submission.csv`

Leaderboard-ready file with two columns:

```csv
id,score
0,0.12
1,0.74
2,0.03
```

- `id`: suspect model id from `0` to `359`
- `score`: continuous stealing-confidence score

### `submission_features.csv`

Full feature table, one row per suspect model. It includes weight similarity, BatchNorm similarity, functional similarity, CKA representation similarity, OOD similarity, augmentation sensitivity, FGSM/Jacobian features, and the final score.

This file is useful for debugging, plotting score distributions, or re-scoring.

### `submission_diagnostics.txt`

Human-readable diagnostics showing:

- top-ranked suspect models
- strongest specialist score per model
- top-20 vs rest feature differences
- possible clean-agreement-only false positives
- possible OOD-only false positives

### `submission_diagnostics.csv`

CSV version of the diagnostics with additional debug columns such as:

```text
dbg_direct_score
dbg_finetune_score
dbg_distilled_score
dbg_boundary_score
dbg_dataset_score
dbg_best_specialist
dbg_clean_only_risk
dbg_ood_only_risk
```

## Method Summary

The detector combines multiple signals:

1. layerwise parameter cosine and sign similarity
2. BatchNorm statistic similarity
3. functional logit similarity on CIFAR-100 test, target-train, and non-target-train samples
4. same-mistake and confident-wrong prediction agreement
5. target-training-subset alignment
6. low-margin and high-entropy target probes
7. synthetic OOD soft-label similarity
8. augmentation and MixMatch-style transform sensitivity
9. linear CKA representation similarity
10. optional FGSM boundary and Jacobian similarity

The final score is a rank-based low-FPR ensemble. It builds specialist scores for direct copies, fine-tuned descendants, distilled/extracted models, boundary-aligned models, and target-training-subset-aligned models. It also penalizes clean-agreement-only and OOD-only patterns to reduce false positives.
