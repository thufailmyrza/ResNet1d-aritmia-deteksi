# Training

This document covers `train/train_model.py`, the `HolterECGDataset` internals (`dataset/holter_dataset.py`), augmentation, monitoring, and a set of pitfalls discovered during development that are easy to reintroduce by accident.

## Contents

- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Dataset Internals](#dataset-internals)
- [Augmentation](#augmentation)
- [Sanity Check](#sanity-check)
- [Checkpointing & Logging](#checkpointing--logging)
- [Monitoring](#monitoring)
- [Known Pitfalls](#known-pitfalls)
- [Fine-Tuning for Target Hardware](#fine-tuning-for-target-hardware)

---

## Quick Start

```bash
# Default (standard model, no resume)
python train/train_model.py

# Recommended: deeper model + SMOTE-balanced training
python train/train_model.py --model-type resnet152 --use-smote --epochs 80

# Resume from best checkpoint
python train/train_model.py --model-type resnet152 --resume
```

Multiprocessing (`--num-workers > 0`) requires the `if __name__ == "__main__":` / `freeze_support()` guard already present in `train_model.py`. On Windows, prefer `--num-workers 0` if you encounter spawn-related errors, and avoid hardcoded absolute paths in `config_path.py` (they break subprocess spawning for DataLoader workers).

---

## CLI Reference

### Model

| Flag | Default | Description |
|---|---|---|
| `--model-type` | `standard` | `standard`, `improved`, or `resnet152` |
| `--dropout` | `0.3` | Dropout rate used in the classification head (and ResBlocks for `standard`/`improved`) |

### Data

| Flag | Default | Description |
|---|---|---|
| `--stride-train` | `500` | Window stride in samples for the training set (500 = 80% overlap for a 2500-sample window) |
| `--use-smote` | `True` | Load synthetic SMOTE windows from `SMOTE_CACHE_DIR`. **Note:** implemented as `action="store_true"` with `default=True`, so it is effectively always `True` regardless of whether the flag is passed — see [Known Pitfalls](#known-pitfalls) |
| `--num-workers` | `4` | DataLoader worker processes (use `0` on Windows if multiprocessing issues occur) |

### Training

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `80` | Max epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-3` | Initial learning rate (AdamW) |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--label-smoothing` | `0.05` | `CrossEntropyLoss` label smoothing (0 = off; values above 0.1 are not recommended) |
| `--grad-clip` | `1.0` | Gradient norm clipping |

### Scheduler

| Flag | Default | Description |
|---|---|---|
| `--scheduler` | `cosine` | `cosine` (`CosineAnnealingWarmRestarts`), `plateau` (`ReduceLROnPlateau` on val Macro-F1), or `step` (`StepLR`, halves every 20 epochs) |
| `--t-max` | `30` | `T_0` for `CosineAnnealingWarmRestarts` |
| `--patience` | `10` | Patience for `ReduceLROnPlateau` |

### Checkpointing

| Flag | Default | Description |
|---|---|---|
| `--resume` | `False` | Resume from `CNN_BEST_MODEL` (optimizer + scheduler state included) |
| `--early-stop-patience` | `20` | Stop if val Macro-F1 doesn't improve for this many epochs |
| `--save-every` | `5` | Save a `last_model.pth` checkpoint every N epochs |

---

## Dataset Internals

`HolterECGDataset` (`dataset/holter_dataset.py`) builds a flat list of `(bin_path, start_offset, class_label, is_real)` entries from a split CSV:

- **Path resolution** (`_resolve_bin_path`) supports both PTB-XL-style rows (`batch_dir` + `output_filename`, resolved against `HOLTER_FORMAT_DIR` or `INCART_FORMAT_DIR` depending on `source`/filename prefix) and INCART/merged-style rows (absolute `filepath`).
- **Minority oversampling**: for non-`oversample_minority=False` datasets, minority classes get a **smaller stride** (more overlapping windows), up to `8×` the base stride reduction, floored at `stride=125`.
- **SMOTE windows**: if `smote_npy_dir` is provided, `synthetic_windows.npy` / `synthetic_labels.npy` are memory-mapped and appended to the index with `is_real=False`.
- **Sampling weights**: `get_sampler()` returns a `WeightedRandomSampler` with inverse-class-frequency weights. **Crucially, frequencies are computed from `window_index[:n_real_windows]` only** — i.e. real windows, before SMOTE entries were appended.
- **Class weights**: `get_class_weights()` similarly uses only real-window counts, normalized so the mean weight is `1.0` (not `sum == num_classes`), which keeps the loss scale stable.

Validation and test datasets are constructed with `stride=2500` (non-overlapping), `augment=False`, `oversample_minority=False`, and `smote_npy_dir=None` — they reflect the true class distribution and real data only.

---

## Augmentation

Applied only when `augment=True` (training set). All operations preserve the mV scale:

| Augmentation | Probability | Effect |
|---|---|---|
| Amplitude scaling | 50% | × `U(0.7, 1.3)` |
| Baseline wander | 50% | + `U(-0.1, 0.1)` mV constant offset |
| Gaussian noise | 50% | + `N(0, 0.02²)` mV per sample |
| Time shift | 50% | Circular shift `±100` samples (`±200 ms`) |
| Lead dropout | 30% | Zero out 1–2 randomly chosen leads |
| Polarity flip | 20% | Invert one randomly chosen lead (simulates reversed electrode) |

---

## Sanity Check

Before the training loop starts, `train_model.py` runs one forward pass on a real batch (random-initialized weights) and checks the initial loss against the theoretical value for a uniform 11-class prediction:

```
expected_loss ≈ ln(11) ≈ 2.398
```

If the observed loss is `< 0.5` or `> 10.0`, a warning is printed — this almost always indicates a **data/label/model bug** (wrong label range, corrupted input, or a precision issue — see [Known Pitfalls](#known-pitfalls)) rather than a training-dynamics issue, since the model has not yet learned anything.

---

## Checkpointing & Logging

- `CNN_BEST_MODEL` (`best_model.pth`) — saved whenever validation Macro-F1 improves; includes `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_macro_f1`.
- `CNN_LAST_MODEL` (`last_model.pth`) — saved every `--save-every` epochs regardless of improvement.
- `training_log.json` — one record per epoch: `loss`, `accuracy`, `macro_f1`, `weighted_f1`, `per_class_f1` (11 values) for both `train` and `val`, plus `lr` and `elapsed_s`. A fresh log is created when **not** resuming (or when no log exists yet) to avoid mixing runs.
- `test_results.json` — written once at the end of training, after reloading `CNN_BEST_MODEL` and evaluating on the test split (includes a full `classification_report` and confusion matrix printed to stdout).

---

## Monitoring

```bash
python tools/plot_training.py --log OUTPUT/checkpoints/cnn/training_log.json --save training_results.png
```

Produces a 6-panel figure: train/val loss, Macro-F1, accuracy, weighted F1, LR schedule, and per-class F1 at the best epoch.

```bash
python tools/plot_model.py --log OUTPUT/checkpoints/cnn/training_log.json --save architecture_holter_ecg.png
```

Produces architecture diagrams (standard + improved variants, SE-ResBlock detail, inference pipeline, class table) alongside training curves if a log is provided.

---

## Known Pitfalls

These are documented because each one produces **plausible-looking but wrong** training behavior — worth checking first if results look off.

### 1. AMP on CPU corrupts logits

Automatic Mixed Precision with `bfloat16` on CPU can silently corrupt logits — `argmax` collapses to class 0 for every sample, and the reported loss looks impossibly low. **Disable AMP (`use_amp=False`) whenever `torch.cuda.is_available()` is `False`.**

### 2. Double-weighting (sampler + loss weight)

Using `WeightedRandomSampler` **and** `CrossEntropyLoss(weight=...)` at the same time applies inverse-frequency correction twice, producing conflicting gradients that prevent convergence. Pick **one**:
- `train_loader = make_loader(train_ds, sampler=train_ds.get_sampler(), ...)` **with** `criterion = nn.CrossEntropyLoss(label_smoothing=...)` (no `weight=`) — this is the current default, **or**
- `shuffle=True` (no sampler) **with** `criterion = nn.CrossEntropyLoss(weight=train_ds.get_class_weights(), ...)`.

### 3. SMOTE contamination of class statistics

`get_class_weights()` and `get_sampler()` must be computed from `n_real_windows` only (see [Dataset Internals](#dataset-internals)). If synthetic SMOTE windows are included in these counts, Normal appears artificially rare (since SMOTE adds tens of thousands of synthetic minority windows), inflating its weight and destabilizing the loss.

### 4. `--use-smote` cannot currently be disabled via CLI

`p.add_argument('--use-smote', default=True, action='store_true', ...)` means the flag is `True` whether or not `--use-smote` is passed. To train **without** SMOTE, either edit this line to `action='store_false'` with an appropriately renamed flag (e.g. `--no-smote`), or pass `smote_npy_dir=None` directly when constructing `HolterECGDataset` in a custom script.

### 5. AF false positives from PVCs

Frequent PVCs cause RR-interval irregularity that can resemble AF. The INCART labeling logic guards against this with a `veb_frac < 0.20` threshold (see [DATASET.md](DATASET.md)); the same caution applies when interpreting AF predictions in NSVT-heavy recordings.

### 6. Windows multiprocessing

`DataLoader(num_workers > 0)` on Windows requires the `if __name__ == '__main__':` guard and `multiprocessing.freeze_support()` (both present in `train_model.py`'s entrypoint). A hardcoded absolute `PROJECT_ROOT` in `config_path.py` can break subprocess spawning — prefer a relative or environment-variable-derived root (see the main [README](../README.md#configuration)).

---

## Fine-Tuning for Target Hardware

Because the Xirka Smart Holter device has its own electrode placement and noise characteristics (different from PTB-XL/INCART acquisition setups), a short fine-tuning pass on real device recordings is recommended before final deployment. The repository includes a fine-tuning script that:

- **Freezes the backbone** and trains only the classification head, using a low learning rate (`AdamW`, `lr=1e-5`).
- Applies **Holter-specific augmentation**: baseline wander, 50 Hz powerline noise, EMG (muscle) noise, and lead dropout — closer to real device noise than the synthetic augmentations used during base training.
- **Boosts class weights** for classes that are clinically important and historically under-detected on device data: Tachycardia (×3.0), Atrial Fibrillation (×3.0), Quadrigeminy (×5.0).
- Uses `ReduceLROnPlateau` and early stopping on a small held-out set of device recordings.

This fine-tuning step is most effective **after** applying the inference-pipeline fixes described in [DEPLOYMENT.md](DEPLOYMENT.md) (ADC gain correction and bandpass filtering) — fine-tuning on top of a domain-shifted input distribution will mask, rather than fix, the underlying mismatch.
