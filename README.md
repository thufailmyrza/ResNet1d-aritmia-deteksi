# ResNet1D-Aritmia-Deteksi

**AI-powered arrhythmia detection for 12-lead ECG Holter recordings**, built with PyTorch and designed for production deployment via ONNX Runtime into the **Xirka Smart Holter (XSH)** desktop application.

This repository contains the full training pipeline — dataset preprocessing, model architectures, training, evaluation, and ONNX export — for a single-label 11-class arrhythmia classifier operating on 5-second windows of 12-lead ECG data.

> Companion application: [APLIKASI-HOLTER-EKG](#) — the PyQt5 desktop viewer that consumes the exported ONNX model produced by this repository.

---

## Table of Contents

- [Overview](#overview)
- [Class Mapping](#class-mapping)
- [Architecture](#architecture)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Known Limitations & Roadmap](#known-limitations--roadmap)
- [Datasets & References](#datasets--references)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

The model classifies 5-second windows of 12-lead ECG (`12 × 2500 samples @ 500 Hz`, float32, millivolts) into one of 11 mutually exclusive classes — normal rhythm or one of ten arrhythmia patterns. Predictions are encoded as bit-flags (`1 << class_index`) so they can be written directly into an `arrhythmia.bin` stream consumed by the Holter viewer application.

Key features:

- **Single-label, 11-class** classification via `CrossEntropyLoss` (not multi-label).
- **Three model variants** sharing the same I/O contract — pick the right trade-off between size and accuracy:
  - `standard` — SE-ResNet-1D, ~2.1M parameters
  - `improved` — deeper SE-ResNet-1D, ~15M parameters
  - `resnet152` — Bottleneck ResNet-1D, ~25M parameters (**currently the deployed variant**)
- **Multi-source training data**: [PTB-XL](https://physionet.org/content/ptb-xl/) (primary, ~21,800 recordings) + [St. Petersburg INCART](https://physionet.org/content/incartdb/1.0.0/) (75 recordings) for classes that are rare or absent in PTB-XL.
- **SMOTE + morphological synthesis** (via Incremental PCA) to balance minority classes that have zero real samples in PTB-XL (Quadrigeminy, Couplet, Triplet).
- **ONNX export pipeline** producing a single-file `.onnx` model that is a drop-in replacement for the detector used by the Holter application, served via `CUDAExecutionProvider` / `CPUExecutionProvider`.

---

## Class Mapping

The model output is `(batch, 11)` logits. `argmax` gives the predicted `class_index`, which is encoded as a bit-flag (`1 << class_index`) for the `arrhythmia.bin` format consumed by the viewer.

| Index | Class                 | Flag (`1 << idx`) |
|:-----:|-----------------------|:------------------:|
| 0     | Normal                | 1                  |
| 1     | Premature Beat        | 2                  |
| 2     | Bigeminy              | 4                  |
| 3     | Trigeminy             | 8                  |
| 4     | Quadrigeminy          | 16                 |
| 5     | Couplet               | 32                 |
| 6     | Triplet               | 64                 |
| 7     | NSVT                  | 128                |
| 8     | Tachycardia           | 256                |
| 9     | Bradycardia           | 512                |
| 10    | Atrial Fibrillation   | 1024               |

> Class 0 (Normal) corresponds to flag `1` (`log2(1) = 0`) and is excluded by the application's region parser — only classes 1–10 are rendered as arrhythmia regions.

---

## Architecture

All three variants share the same input/output contract:

```
Input  : (batch, 12, 2500) float32, millivolts, 500 Hz, 5-second window
Output : (batch, 11) raw logits  →  argmax  →  class_index (0–10)
```

The default (`standard`) architecture is a 1D ResNet with Squeeze-and-Excitation channel attention and multi-scale temporal attention:

```
Input (B, 12, 2500)
   │
   ▼
Stem: Conv1D(k=15, s=2) → BN → ReLU → MaxPool      → (B, 64, 1250)
   │
   ▼
Stage 1: 2× SE-ResBlock(64→64,   s=1)              → (B, 64, 1250)
Stage 2: 2× SE-ResBlock(64→128,  s=2)              → (B, 128, 625)
Stage 3: 2× SE-ResBlock(128→256, s=2)              → (B, 256, 313)
Stage 4: 2× SE-ResBlock(256→512, s=2)              → (B, 512, 157)
   │
   ▼
Multi-Scale Temporal Attention                      → (B, 512, 157)
   │
   ▼
Global Average Pool                                 → (B, 512)
   │
   ▼
Head: Linear(512→256) → BN → ReLU → Dropout → Linear(256→11)
   │
   ▼
Output (B, 11) logits
```

The `improved` variant uses a deeper `[3, 4, 6, 3]` SE-ResBlock layout, and `resnet152` uses Bottleneck blocks (1×1 → 3×3 → 1×1 with SE) following a `[3, 4, 6, 3]` stage layout with channels scaling up to 2048.

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for full details, including the SE-ResBlock and Bottleneck block diagrams, parameter counts, and the inference → `arrhythmia.bin` pipeline.

---

## Results

The currently deployed checkpoint is the **`resnet152`** variant, selected at **epoch 63** with **Macro F1 = 0.8499** on the held-out test split (PTB-XL + INCART merged dataset).

Performance is not uniform across classes — common classes (Normal, Premature Beat, AF, Tachycardia/Bradycardia) perform well, while classes with no real training examples (Quadrigeminy, Couplet, Triplet — synthesized via morphological interpolation + SMOTE) remain harder. See **[docs/TRAINING.md](docs/TRAINING.md)** for per-class breakdowns, training curves, and the fine-tuning workflow used to adapt the model to the target Holter hardware.

---

## Repository Structure

```
ResNet1d-aritmia-deteksi/
├── README.md
├── requirements.txt
├── config_path.py            # Central path & constant configuration
│
├── model/
│   ├── resnet1d.py            # standard + improved variants, build_model() factory
│   ├── resnet152.py            # resnet152 (Bottleneck) variant
│   └── smote_oversampling.py  # PCA + SMOTE + morphological synthesis
│
├── dataset/
│   ├── preprocess_ptbxl.py     # PTB-XL → binary windows + labels.csv + splits
│   ├── convert_incart.py       # INCART → binary windows + labels.csv
│   ├── merge_dataset.py        # Merge PTB-XL + INCART, build final splits
│   └── holter_dataset.py       # PyTorch Dataset (windowing, sampling, augmentation)
│
├── train/
│   ├── train_model.py          # Training entrypoint
│   └── export_model.py         # ONNX / PKL export + inference utilities
│
├── tools/
│   ├── plot_training.py        # Training curve visualization
│   ├── plot_model.py           # Architecture & pipeline diagrams
│   ├── fix_export.py           # Single-file ONNX export (merges external data)
│   ├── verify_onnx.py          # ONNX Runtime sanity check
│   └── diagnostics/
│       ├── verify_cuda.py      # CUDA / torch availability check
│       ├── diagnose_incart.py  # INCART dataset diagnostics
│       └── probe_xirka.py      # Inspect raw Xirka device .bin files
│
└── docs/
    ├── ARCHITECTURE.md
    ├── DATASET.md
    ├── TRAINING.md
    └── DEPLOYMENT.md
```

> **Note for contributors migrating from the working copy:** the original project keeps several utility scripts (`plot_training.py`, `plot_model.py`, `fix_export.py`, `verify_onnx.py`) and a folder named `error check/` at the project root. For the public layout above, these were grouped under `tools/` and `tools/diagnostics/` respectively (the space in `error check/` is intentionally avoided — it breaks Python module imports and some shells). Update any hardcoded relative imports accordingly.

---

## Getting Started

### Requirements

- Python 3.12
- A CUDA-capable GPU is **strongly recommended** for training (CPU training works but is slow, and AMP/mixed precision is automatically disabled on CPU — see [docs/TRAINING.md](docs/TRAINING.md)).
- For ONNX export verification with GPU: a working `onnxruntime-gpu` install matched to your installed CUDA Toolkit version. Check compatibility with `tools/diagnostics/verify_cuda.py` and `tools/verify_onnx.py` before relying on `CUDAExecutionProvider`.

### Installation

```bash
git clone https://github.com/thufailmyrza/ResNet1d-aritmia-deteksi.git
cd ResNet1d-aritmia-deteksi

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
```

### Configuration

All paths and dataset/model constants are centralized in **`config_path.py`**. The shipped version uses an absolute Windows development path (`PROJECT_ROOT = Path("C:/Users/.../Project Arrythmia")`) — **update this before running anything**, or replace it with an environment-variable-driven root so the repo works on any machine:

```python
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("ARITMIA_PROJECT_ROOT", Path(__file__).resolve().parent))
```

Then set `ARITMIA_PROJECT_ROOT` (or edit the path directly) to point at the folder that will contain `RAW DATA/`, `OUTPUT/`, and checkpoints.

---

## Quick Start

### 1. Prepare the datasets

Download [PTB-XL](https://physionet.org/content/ptb-xl/) and the [St. Petersburg INCART](https://physionet.org/content/incartdb/1.0.0/) database from PhysioNet and place them according to `config_path.py` (`RAW DATA/ptb-xl`, `RAW DATA/incart`, including `record-descriptions.txt` for INCART). Then run:

```bash
python dataset/preprocess_ptbxl.py     # → binary windows, labels.csv, train/val/test splits
python dataset/convert_incart.py       # → INCART binary windows + incart_labels.csv
python dataset/merge_dataset.py        # → merges both sources, overwrites train/val/test splits
python model/smote_oversampling.py     # optional: builds synthetic_windows.npy / synthetic_labels.npy
```

Details on the labeling logic, class resolution, and file formats are in **[docs/DATASET.md](docs/DATASET.md)**.

### 2. Train

```bash
python train/train_model.py --model-type resnet152 --use-smote --epochs 80
```

Checkpoints, `training_log.json`, and `test_results.json` are written under `OUTPUT/checkpoints/cnn/`. Full CLI reference, augmentation details, and important training pitfalls (e.g. CPU/AMP issues, SMOTE-aware class weighting) are documented in **[docs/TRAINING.md](docs/TRAINING.md)**.

### 3. Visualize

```bash
python tools/plot_training.py --log OUTPUT/checkpoints/cnn/training_log.json --save training_results.png
python tools/plot_model.py --log OUTPUT/checkpoints/cnn/training_log.json --save architecture_holter_ecg.png
```

### 4. Export to ONNX

```bash
python train/export_model.py export --model-type resnet152 --checkpoint OUTPUT/checkpoints/cnn/best_model.pth
```

For a single-file `.onnx` (no external `.onnx.data` companion file), use:

```bash
python tools/fix_export.py
```

### 5. Run inference on a recording

```bash
python train/export_model.py infer \
  --model OUTPUT/exported_models/arrhythmia_model.onnx \
  --input path/to/recording.bin \
  --source device
```

This produces an `arrhythmia.bin` file in the same format consumed by the Holter viewer's `arrhythmia_parser.py`.

### 6. Integrate with the Xirka Holter App

Copy the exported `.onnx` file into the application's `resource/models/detector.onnx`. The application's `ArrhythmiaDetector` class loads it via `onnxruntime` (`CUDAExecutionProvider` → `CPUExecutionProvider` fallback) and writes `arr.bin` next to the recording. The full integration contract, validation results on real device recordings, and required fixes on the application side are documented in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Model variants, SE-ResBlock / Bottleneck block details, parameter counts, full inference pipeline |
| [docs/DATASET.md](docs/DATASET.md) | PTB-XL & INCART preprocessing, label resolution, SMOTE/morphological synthesis, binary file formats |
| [docs/TRAINING.md](docs/TRAINING.md) | Training CLI reference, dataset/sampler internals, augmentation, known pitfalls, fine-tuning |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | ONNX export & runtime contract, integration with the Holter app, validation results, known issues |

---

## Known Limitations & Roadmap

- **Quadrigeminy** has zero real training samples in either source dataset; it is built entirely from morphological interpolation + SMOTE and currently has the weakest F1 of all classes.
- **AF / Tachycardia / Bradycardia** can be confused in edge cases where RR-interval irregularity from frequent PVCs mimics atrial fibrillation.
- **Hardware domain gap**: leads III and aVF on the Xirka device (torso electrode placement) show systematically lower amplitude than the training data; partially mitigated by ±30% amplitude augmentation, but a hardware-specific fine-tuning pass on real device recordings is recommended (see [docs/TRAINING.md](docs/TRAINING.md) and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)).
- The training CLI's `--use-smote` flag currently defaults to `True` with `action="store_true"`, so it cannot be disabled from the command line — see [docs/TRAINING.md](docs/TRAINING.md) for the workaround.
- `train/export_model.py`'s `export` subcommand does not list `resnet152` as a valid `--model-type` choice (use `tools/fix_export.py`, or pass `model_type="resnet152"` directly to `export_pipeline()`).

Contributions addressing any of the above are welcome.

---

## Datasets & References

- Wagner, P., Strodthoff, N., Bousseljot, R., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020). **PTB-XL, a large publicly available electrocardiography dataset.** *Scientific Data*. Hosted on [PhysioNet](https://physionet.org/content/ptb-xl/).
- **St. Petersburg INCART 12-lead Arrhythmia Database.** Hosted on [PhysioNet](https://physionet.org/content/incartdb/1.0.0/).

If you use this repository, please also cite the above dataset sources per PhysioNet's terms of use.

---

## License

*This section is a placeholder — add a `LICENSE` file before publishing.* For research code built on PTB-XL and INCART (both distributed under the [Open Data Commons Attribution License](https://physionet.org/content/ptb-xl/view-license/1.0.3/)), a permissive license such as **MIT** or **Apache-2.0** is common; ensure your chosen license is compatible with the dataset licenses if you redistribute any derived data artifacts (synthetic windows, preprocessed binaries, etc. should generally **not** be redistributed).

---

## Acknowledgments

- **Xirka** — hardware partner providing the Smart Holter (XSH) device and the host Holter viewer application.
- **PhysioNet / PTB-XL / St. Petersburg INCART** contributors for the open datasets that make this project possible.
