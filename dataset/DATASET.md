# Dataset

This document describes the data sources, preprocessing pipelines, label resolution logic, binary file formats, and the SMOTE/morphological synthesis used to balance minority classes.

## Contents

- [Data Sources](#data-sources)
- [Common Conventions](#common-conventions)
- [PTB-XL Preprocessing](#ptb-xl-preprocessing-datasetpreprocess_ptbxlpy)
- [INCART Preprocessing](#incart-preprocessing-datasetconvert_incartpy)
- [Merging Datasets](#merging-datasets-datasetmerge_datasetpy)
- [SMOTE & Morphological Synthesis](#smote--morphological-synthesis-modelsmote_oversamplingpy)
- [Final Directory Layout](#final-directory-layout)
- [Reproducing the Full Pipeline](#reproducing-the-full-pipeline)

---

## Data Sources

| Source | Role | Size | Sampling rate | Notes |
|---|---|---|---|---|
| [PTB-XL](https://physionet.org/content/ptb-xl/) | Primary | ~21,799 recordings | 500 Hz (`records500`) | Multi-label SCP codes; resolved to single-label via priority order |
| [St. Petersburg INCART](https://physionet.org/content/incartdb/1.0.0/) | Supplementary | 75 recordings | 257 Hz → resampled to 500 Hz | Provides classes absent or rare in PTB-XL (Quadrigeminy, Couplet, Triplet, NSVT); labeled via `record-descriptions.txt` + beat annotations |

Both sources are converted into the **same binary window format** so that `dataset/holter_dataset.py` can consume them transparently.

---

## Common Conventions

- **Target sampling rate**: 500 Hz (`HOLTER_SAMPLING_RATE`)
- **Lead order**: `I, II, III, aVR, aVF, aVL, V1–V6` (`ECG_CHANNELS`, 12 channels)
- **Window size**: 2500 samples = 5 seconds (`WINDOW_SIZE`)
- **Bandpass filter**: 4th-order Butterworth, 0.5–40 Hz, applied with `filtfilt` (zero-phase) before windowing
- **On-disk format**: each window is stored as raw `int16`, shape `(2500, 12)`, row-major (sample-major, channel-minor) — i.e. `(window_size × NUM_CHANNELS)` int16 values per file
- **Scale conversion**:
  - PTB-XL / merged binaries: `mV = int16_value × INT16_TO_MV` where `INT16_TO_MV = 1/1000 = 0.001`
  - Raw Xirka device binaries: `mV = int16_value × ADC_GAIN_DEVICE` where `ADC_GAIN_DEVICE = 0.0025`

All of the above constants live in `config_path.py` and are shared across preprocessing, training, and export.

---

## PTB-XL Preprocessing (`dataset/preprocess_ptbxl.py`)

1. **Load metadata** from `ptbxl_database.csv`.
2. **Parse SCP codes** (`scp_codes` column, a stringified dict) into a list of codes per recording.
3. **Map SCP codes → class indices** via `PTBXL_TO_CLASS`, e.g.:
   - `PVC`, `VPVC`, `SVPB`, `PAC`, `SVARR`, `EL` → **Premature Beat** (1)
   - `BIGU` → **Bigeminy** (2), `TRIGU` → **Trigeminy** (3)
   - `STACH`, `SVTAC`, `SVT`, `PSVT`, `AVNRT`, `AVRT`, `AT` → **Tachycardia** (8)
   - `SBRAD` → **Bradycardia** (9)
   - `AFIB`, `AF` → **Atrial Fibrillation** (10)
   - Codes with no mapping (`SARRH`, `AFLT`, `I-AVB`, `II-AVB`, `III-AVB`, etc.) are dropped — a recording with **only** these codes resolves to **Normal** (0).
   - Note: PTB-XL has **no explicit codes** for Quadrigeminy, Couplet, Triplet, or NSVT — these classes receive zero real samples from PTB-XL and rely entirely on INCART + synthetic data (see below).
4. **Resolve multi-label → single-label** via `ARRHYTHMIA_PRIORITY` (most specific/severe first):
   ```
   atrial_fibrillation > bradycardia > tachycardia > nsvt > triplet > couplet
     > quadrigeminy > trigeminy > bigeminy > premature_beat > normal
   ```
   The first class in this list present in the recording's class set becomes `class_label`. The full multi-label set is also preserved as `arrhythmia_bitmask` for reference/analysis.
5. **Signal processing**: resample to 500 Hz if needed, reorder leads to `ECG_CHANNELS`, apply the 0.5–40 Hz bandpass filter, convert to `int16` (`mV × 1000`, clipped to `int16` range).
6. **Write output**:
   - One `.bin` file per recording under `HOLTER_FORMAT_DIR/batch_XXXXX/`
   - `labels.csv` with columns including `output_filename`, `batch_dir`, `class_label`, `arrhythmia_bitmask`, `has_arrhythmia`, `scp_codes`, `age`, `sex`, `report`, `success`
   - `dataset_statistics.json` with per-class counts
7. **Train/val/test split**: stratified by `class_label` (or `has_arrhythmia` if any class has fewer than 4 samples), default ratios `0.80 / 0.10 / 0.10`, written to `train_split.csv`, `val_split.csv`, `test_split.csv`.

---

## INCART Preprocessing (`dataset/convert_incart.py`)

INCART recordings are long (~30 min) and are split into many overlapping windows. Labeling happens at two levels:

### Level 1 — Per-recording primary class

`record-descriptions.txt` contains a free-text description per recording (e.g. *"ventricular trigeminy, ventricular couplets"*). A keyword map (`_KEYWORD_MAP`, regex-based) extracts a **set** of candidate classes per recording, which is resolved to a single **primary class** using the same `ARRHYTHMIA_PRIORITY` order as PTB-XL.

### Level 2 — Per-window activity detection

For each 2500-sample window (stride 500 samples = 1 s), beat annotations (`.atr` files, rescaled from 257 Hz to 500 Hz) determine whether the window is **active** (label = primary class) or **normal** (label = 0):

| Primary class | Activity rule |
|---|---|
| Premature Beat, Bigeminy, Trigeminy, Quadrigeminy, Couplet, Triplet, NSVT | Active if **any** ectopic beat (`V`, `E`, `F`, `S`, `A`, `a`, `J`, `j`, `e`) is present in the window |
| Tachycardia | Active if mean HR from beats in the window ≥ 100 bpm |
| Bradycardia | Active if mean HR ≤ 60 bpm |
| Atrial Fibrillation | Active if RR-interval coefficient of variation > 0.20 **and** ventricular-ectopic-beat fraction < 0.20 (guards against PVC-driven irregularity being misread as AF) |
| Normal (record I60) | Always inactive (label 0) |

Each window is bandpass-filtered (same 0.5–40 Hz filter), resampled to 500 Hz, reordered to the standard lead order, converted to `int16`, and written to `INCART_FORMAT_DIR/batch_XXXXX/incart_NNNNNN.bin`, with one row per window in `incart_labels.csv` (columns: `filepath`, `class_index`, `class_name`, `has_arrhythmia`, `source='incart'`, `record_name`, `win_start_sec`, `success`).

> **wfdb quirk**: `wfdb.rdsamp` returns INCART signals already in millivolts (no manual ADC-gain division needed), and lead names use `AVR`/`AVL`/`AVF` (uppercase, no accent) rather than `aVR`/`aVL`/`aVF`.

---

## Merging Datasets (`dataset/merge_dataset.py`)

1. **Load PTB-XL** labels and resolve the legacy `arrhythmia_bitmask` (a different bit layout than the current 11-class scheme — see `PTBXL_OLD_BIT_TO_CLASS`) to a `class_index` via `ARRHYTHMIA_PRIORITY`.
2. **Load INCART** labels (already single-label) and normalize column names (`filepath` → derive `batch_dir` / `output_filename`; `class_index` → alias `class_label`) so `holter_dataset.py` can read both sources uniformly.
3. **Concatenate** both DataFrames, aligning columns, filling `class_index`/`class_label` from each other, and recomputing `has_arrhythmia` / `class_name`.
4. **Stratified 3-way split** using a custom stratification key:
   - **Stratum 0** — Normal
   - **Stratum 1** — "old" arrhythmia classes (present in PTB-XL)
   - **Stratum 2** — "new" classes from INCART only (Quadrigeminy, Couplet, Triplet, NSVT)

   This guarantees the rare INCART-only classes are represented in train/val/test, using ratios `0.75 / 0.15 / 0.10`.
5. **Overwrite** `train_split.csv`, `val_split.csv`, `test_split.csv` in `HOLTER_FORMAT_DIR` — downstream training code requires no changes. Also writes `merged_labels.csv` and `merged_statistics.json`.

---

## SMOTE & Morphological Synthesis (`model/smote_oversampling.py`)

Even after merging, three classes (**Quadrigeminy**, **Couplet**, **Triplet**) have **zero real windows** in the training set. The synthesis pipeline:

1. **Extract windows** from the training split (non-overlapping, capped at 5,000 windows/class) → `(N, 12, 2500)` float32 mV.
2. **Morphological synthesis** for the zero-data classes — generate synthetic windows by interpolating between **structurally similar** real classes:
   - Quadrigeminy (4) ← interpolation of Bigeminy (2) + Trigeminy (3), `α ~ U(0.3, 0.7)`
   - Couplet (5) ← interpolation of Premature Beat (1) + Bigeminy (2), `α ~ U(0.4, 0.8)`
   - Triplet (6) ← interpolation of Premature Beat (1) + Bigeminy (2), `α ~ U(0.2, 0.6)`, plus small Gaussian noise (σ = 0.01 mV)
   - 2,000 synthetic windows per class by default.
3. **Fit `IncrementalPCA`** (default 256 components) on the combined real + morphological windows, flattened to `(N, 12 × 2500)`.
4. **SMOTE / BorderlineSMOTE** in PCA space, targeting each minority class up to the majority class count (or a fixed target for the zero-data classes).
5. **Inverse-PCA** the synthetic PCA vectors back to `(N_syn, 12, 2500)` mV windows.
6. **Save** `synthetic_windows.npy`, `synthetic_labels.npy`, and `smote_stats.json` to `SMOTE_CACHE_DIR`.

`HolterECGDataset` loads these synthetic windows alongside real windows when `smote_npy_dir` is provided, but — critically — **class weights and the `WeightedRandomSampler` are computed from real windows only** (`n_real_windows`). Mixing synthetic counts into those statistics would make Normal appear artificially rare and destabilize training (see [TRAINING.md](TRAINING.md) for details).

---

## Final Directory Layout

```
OUTPUT/
├── HOLTER_V5/                      # HOLTER_FORMAT_DIR (PTB-XL + merged)
│   ├── batch_00000/ ... batch_NNNNN/
│   │   └── <record>.bin            # (2500, 12) int16 per window, row-major
│   ├── labels.csv                  # PTB-XL only
│   ├── merged_labels.csv           # PTB-XL + INCART
│   ├── merged_statistics.json
│   ├── dataset_statistics.json
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   └── smote_cache/
│       ├── synthetic_windows.npy
│       ├── synthetic_labels.npy
│       └── smote_stats.json
│
└── INCART_FORMAT/                  # INCART_FORMAT_DIR
    ├── batch_00000/ ... 
    │   └── incart_NNNNNN.bin
    ├── incart_labels.csv
    └── incart_statistics.json
```

---

## Reproducing the Full Pipeline

```bash
# 1. Place raw data per config_path.py:
#    RAW DATA/ptb-xl/...               (PTB-XL download, including records500/)
#    RAW DATA/incart/...               (INCART download + record-descriptions.txt + RECORDS)

# 2. Preprocess PTB-XL → binary windows, labels.csv, initial train/val/test splits
python dataset/preprocess_ptbxl.py

# 3. Preprocess INCART → binary windows, incart_labels.csv
python dataset/convert_incart.py
# (optional dry run on a single record first)
python dataset/convert_incart.py --dry-run --verbose

# 4. Merge both sources → overwrites train/val/test splits with stratified 3-strata split
python dataset/merge_dataset.py

# 5. (optional) Build synthetic windows for zero-data classes
python model/smote_oversampling.py --n-pca 256 --morph-n 2000
```

After step 5, `train/train_model.py --use-smote` will pick up `synthetic_windows.npy` / `synthetic_labels.npy` automatically from `SMOTE_CACHE_DIR`.
