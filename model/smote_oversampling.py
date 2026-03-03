"""
smote_oversampling.py  –  REVISI v3 (baru)
SMOTE untuk kelas minoritas pada windowed ECG data.

Pendekatan:
  1. Ekstrak semua window dari training set → (N, 12, 2500)
  2. Flatten → (N, 30000)
  3. PCA reduction → (N, n_components) – mengurangi dimensi agar SMOTE feasible
  4. SMOTE per kelas minoritas di ruang PCA
  5. Inverse PCA → (N_synthetic, 12, 2500)  [dalam satuan mV]
  6. Simpan synthetic_windows.npy + synthetic_labels.npy

Kelas tanpa data di PTB-XL (class 4=quadrigeminy, 5=couplet, 6=triplet)
ditangani dengan "Morphological SMOTE" – interpolasi antara kelas bigeminy
dan trigeminy yang strukturnya paling mirip.

Output:
  SMOTE_CACHE_DIR/
    synthetic_windows.npy   → (N_syn, 12, 2500) float32 (mV)
    synthetic_labels.npy    → (N_syn,) int64
    smote_stats.json        → ringkasan per kelas
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import sys
import warnings

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config_path import (HOLTER_FORMAT_DIR, TRAIN_SPLIT_CSV, SMOTE_CACHE_DIR, 
                         NUM_ARRHYTHMIA_CLASSES, NUM_CHANNELS, ARRHYTHMIA_CLASSES,
                         INT16_TO_MV, WINDOW_SIZE,
                         )

warnings.filterwarnings('ignore')

# ── Konfigurasi ──────────────────────────────────────────────────────────────

# Target jumlah sample per kelas setelah SMOTE
# (None = samakan dengan kelas mayoritas)
TARGET_PER_CLASS: dict[int, int | None] = {
    0:  None,    # normal       → ikuti mayoritas
    1:  None,    # premature    → ikuti mayoritas
    2:  None,    # bigeminy
    3:  None,    # trigeminy
    4:  2000,    # quadrigeminy → 0 PTB-XL, morfologis dari cls 2+3
    5:  2000,    # couplet      → 0 PTB-XL, morfologis dari cls 1+2
    6:  2000,    # triplet      → 0 PTB-XL, morfologis dari cls 1+2
    7:  None,    # nsvt
    8:  None,    # tachycardia
    9:  None,    # bradycardia
    10: None,    # AF
}

# Kelas sumber untuk kelas tanpa data PTB-XL
MORPHOLOGICAL_SOURCE: dict[int, list[int]] = {
    4: [2, 3],   # quadrigeminy ← bigeminy + trigeminy
    5: [1, 2],   # couplet      ← premature + bigeminy
    6: [1, 2],   # triplet      ← premature + bigeminy
}

# PCA components
N_PCA_COMPONENTS = 256

# Window stride saat ekstraksi (lebih besar = lebih cepat, lebih sedikit memori)
EXTRACTION_STRIDE = 2500   # non-overlapping untuk efisiensi

# Batas maksimum window yang di-load (untuk keterbatasan RAM)
MAX_WINDOWS_PER_CLASS = 5000


# ============================================================================
# STEP 1: EKSTRAKSI WINDOWS
# ============================================================================

def extract_windows(labels_csv: Path, data_root: Path,
                    window_size: int = WINDOW_SIZE,
                    stride: int = EXTRACTION_STRIDE,
                    max_per_class: int = MAX_WINDOWS_PER_CLASS
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Ekstrak window ECG dari training set.

    Returns:
        windows : (N, 12, window_size) float32 (mV)
        labels  : (N,) int64 (class index 0–10)
    """
    print("=" * 70)
    print("STEP 1: EKSTRAKSI WINDOWS")
    print("=" * 70)

    df       = pd.read_csv(labels_csv)
    df       = df[df.get('success', True) == True].reset_index(drop=True)
    data_dir = Path(data_root)

    windows_per_cls: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_ARRHYTHMIA_CLASSES)}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ekstraksi"):
        cls      = int(row['class_label'])
        bin_path = data_dir / row['batch_dir'] / row['output_filename']

        if not bin_path.exists():
            continue
        if len(windows_per_cls[cls]) >= max_per_class:
            continue

        n_total  = bin_path.stat().st_size // (NUM_CHANNELS * 2)
        n_win    = max(1, (n_total - window_size) // stride + 1)

        with open(bin_path, 'rb') as f:
            for w in range(n_win):
                if len(windows_per_cls[cls]) >= max_per_class:
                    break
                start = w * stride
                f.seek(start * NUM_CHANNELS * 2)
                raw = np.fromfile(f, dtype=np.int16,
                                  count=window_size * NUM_CHANNELS)
                if len(raw) < window_size * NUM_CHANNELS:
                    raw = np.pad(raw,
                                 (0, window_size * NUM_CHANNELS - len(raw)))
                ecg_mv = (raw.reshape(-1, NUM_CHANNELS).T.astype(np.float32)
                          * INT16_TO_MV)   # (12, W) mV
                windows_per_cls[cls].append(ecg_mv)

    # Gabungkan
    all_wins, all_labs = [], []
    for cls, wins in windows_per_cls.items():
        if wins:
            arr  = np.stack(wins, axis=0)   # (N_cls, 12, W)
            all_wins.append(arr)
            all_labs.append(np.full(len(wins), cls, dtype=np.int64))
            print(f"  Class {cls:2d} [{ARRHYTHMIA_CLASSES[cls]:22s}]: {len(wins):,} windows")

    windows = np.concatenate(all_wins, axis=0)
    labels  = np.concatenate(all_labs, axis=0)
    print(f"\nTotal windows: {len(windows):,}")
    return windows, labels


# ============================================================================
# STEP 2: PCA REDUCTION
# ============================================================================

def fit_pca(windows: np.ndarray, n_components: int = N_PCA_COMPONENTS,
            batch_size: int = 512) -> IncrementalPCA:
    """
    Fit IncrementalPCA pada windows (12×W → n_components).
    Incremental agar tidak memerlukan semua data di RAM sekaligus.

    Args:
        windows    : (N, 12, W) float32
        n_components: Jumlah PCA components
        batch_size : Batch size per partial_fit

    Returns:
        Fitted IncrementalPCA object
    """
    print("\n" + "=" * 70)
    print(f"STEP 2: FIT PCA  ({n_components} components)")
    print("=" * 70)

    N = len(windows)
    flat = windows.reshape(N, -1)   # (N, 12*W)

    pca = IncrementalPCA(n_components=n_components)
    for start in tqdm(range(0, N, batch_size), desc="PCA fit"):
        pca.partial_fit(flat[start:start + batch_size])

    total_var = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance: {total_var*100:.1f}%")
    return pca


def transform_pca(windows: np.ndarray, pca: IncrementalPCA,
                  batch_size: int = 512) -> np.ndarray:
    """Transform windows ke PCA space. Returns (N, n_components)."""
    N    = len(windows)
    flat = windows.reshape(N, -1)
    out  = []
    for s in range(0, N, batch_size):
        out.append(pca.transform(flat[s:s + batch_size]))
    return np.concatenate(out, axis=0)


def inverse_transform_pca(reduced: np.ndarray, pca: IncrementalPCA,
                           window_shape: tuple,
                           batch_size: int = 512) -> np.ndarray:
    """Inverse PCA → windows shape (N, 12, W) float32."""
    out = []
    for s in range(0, len(reduced), batch_size):
        chunk = pca.inverse_transform(reduced[s:s + batch_size])
        out.append(chunk)
    flat = np.concatenate(out, axis=0)
    return flat.reshape(-1, *window_shape).astype(np.float32)


# ============================================================================
# STEP 3: SMOTE
# ============================================================================

def apply_smote(reduced: np.ndarray, labels: np.ndarray,
                target_per_class: dict[int, int | None]
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Terapkan SMOTE pada data PCA-reduced untuk menyeimbangkan kelas.

    Hanya kelas yang ada data-nya (di PTB-XL) yang di-SMOTE.
    Kelas tanpa data (quadrigeminy, couplet, triplet) ditangani terpisah
    di generate_morphological_samples().

    Args:
        reduced : (N, n_components) float32 – fitur PCA
        labels  : (N,) int64 – class index

    Returns:
        resampled_reduced : (N + N_syn, n_components)
        resampled_labels  : (N + N_syn,) int64
    """
    print("\n" + "=" * 70)
    print("STEP 3: SMOTE")
    print("=" * 70)

    unique, counts = np.unique(labels, return_counts=True)
    print("  Distribusi sebelum SMOTE:")
    for u, c in zip(unique, counts):
        print(f"    Class {u:2d} [{ARRHYTHMIA_CLASSES[u]:22s}]: {c:,}")

    # Tentukan target sampling strategy
    max_count = counts.max()
    strategy  = {}

    for cls_idx, count in zip(unique, counts):
        tgt = target_per_class.get(int(cls_idx))
        if tgt is None:
            tgt = int(max_count)
        if tgt > count:
            strategy[int(cls_idx)] = tgt

    if not strategy:
        print("  Tidak ada kelas yang perlu di-oversample.")
        return reduced, labels

    print(f"\n  Target SMOTE: {strategy}")

    # Gunakan BorderlineSMOTE untuk kelas dengan data sedikit,
    # SMOTE biasa untuk yang lebih banyak
    min_count = min(counts[counts > 0])

    if min_count < 6:
        # k_neighbors harus < min class size
        k = max(1, min_count - 1)
        smote = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=42)
    else:
        smote = BorderlineSMOTE(sampling_strategy=strategy,
                                k_neighbors=5, random_state=42)

    resampled_r, resampled_l = smote.fit_resample(
        reduced.astype(np.float64), labels
    )

    _, new_counts = np.unique(resampled_l, return_counts=True)
    print("\n  Distribusi setelah SMOTE:")
    u2, c2 = np.unique(resampled_l, return_counts=True)
    for u, c in zip(u2, c2):
        print(f"    Class {u:2d} [{ARRHYTHMIA_CLASSES[u]:22s}]: {c:,}")

    return resampled_r.astype(np.float32), resampled_l.astype(np.int64)


# ============================================================================
# STEP 4: GENERATE MORPHOLOGICAL SAMPLES (kelas tanpa data PTB-XL)
# ============================================================================

def generate_morphological_samples(
    existing_windows: np.ndarray,
    existing_labels:  np.ndarray,
    n_per_class: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Buat synthetic windows untuk kelas tanpa data PTB-XL
    (quadrigeminy=4, couplet=5, triplet=6) via interpolasi linear
    antara kelas sumber terdekat.

    Kelas target → sumber:
      4 (quadrigeminy) ← interpolasi bigeminy(2) + trigeminy(3), α~U(0.3,0.7)
      5 (couplet)      ← interpolasi premature(1) + bigeminy(2), α~U(0.4,0.8)
      6 (triplet)      ← interpolasi premature(1) + bigeminy(2), α~U(0.2,0.6)
         + gaussian noise kecil (0.01 mV std)

    Returns:
        windows : (N_new, 12, W) float32 mV
        labels  : (N_new,) int64
    """
    print("\n" + "=" * 70)
    print("STEP 4: MORPHOLOGICAL SAMPLES (kelas tanpa data)")
    print("=" * 70)

    new_wins, new_labs = [], []
    rng = np.random.default_rng(seed=42)

    for tgt_cls, src_classes in MORPHOLOGICAL_SOURCE.items():
        # Kumpulkan window dari kelas sumber
        src_wins_list = []
        for sc in src_classes:
            mask = existing_labels == sc
            if mask.sum() == 0:
                print(f"  ⚠ Class {sc} tidak ada data, kelas {tgt_cls} dilewati.")
                break
            src_wins_list.append(existing_windows[mask])
        else:
            # Semua sumber tersedia → lakukan interpolasi
            print(f"  Generating class {tgt_cls} "
                  f"[{ARRHYTHMIA_CLASSES[tgt_cls]}] "
                  f"dari class {src_classes}…")

            gen_wins = []
            for _ in range(n_per_class):
                # Pilih satu window dari setiap kelas sumber secara acak
                picked = [w[rng.integers(0, len(w))] for w in src_wins_list]

                # Koefisien interpolasi disampling dari simplex
                if len(picked) == 2:
                    alpha = rng.uniform(0.3, 0.7)
                    mixed = alpha * picked[0] + (1 - alpha) * picked[1]
                else:
                    alphas = rng.dirichlet(np.ones(len(picked)))
                    mixed  = sum(a * p for a, p in zip(alphas, picked))

                # Tambahkan noise kecil (0.01 mV std)
                mixed = mixed + rng.normal(0, 0.01, mixed.shape).astype(np.float32)
                gen_wins.append(mixed.astype(np.float32))

            arr = np.stack(gen_wins, axis=0)   # (n_per_class, 12, W)
            new_wins.append(arr)
            new_labs.append(np.full(n_per_class, tgt_cls, dtype=np.int64))
            print(f"    ✓ {n_per_class:,} windows dibuat")

    if not new_wins:
        return np.empty((0, NUM_CHANNELS, WINDOW_SIZE), dtype=np.float32), \
               np.empty((0,), dtype=np.int64)

    return (np.concatenate(new_wins, axis=0),
            np.concatenate(new_labs, axis=0))


# ============================================================================
# PIPELINE UTAMA
# ============================================================================

def run_smote_pipeline(
    labels_csv: Path     = TRAIN_SPLIT_CSV,
    data_root: Path      = HOLTER_FORMAT_DIR,
    output_dir: Path     = SMOTE_CACHE_DIR,
    n_pca: int           = N_PCA_COMPONENTS,
    target: dict         = TARGET_PER_CLASS,
    morph_n: int         = 2000,
) -> None:
    """
    Pipeline lengkap SMOTE untuk training set ECG Holter.

    Output di output_dir/:
      synthetic_windows.npy   (N_syn, 12, 2500) float32 mV
      synthetic_labels.npy    (N_syn,) int64
      smote_stats.json        ringkasan
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SMOTE PIPELINE – ECG Holter (v3)")
    print("=" * 70)

    # ── 1. Ekstraksi windows ──────────────────────────────────────────────
    windows, labels = extract_windows(
        labels_csv=labels_csv,
        data_root=data_root,
    )
    n_original = len(windows)

    # ── 2. Morphological untuk kelas tanpa data ───────────────────────────
    morph_wins, morph_labs = generate_morphological_samples(
        windows, labels, n_per_class=morph_n
    )
    if len(morph_wins) > 0:
        windows = np.concatenate([windows, morph_wins], axis=0)
        labels  = np.concatenate([labels,  morph_labs],  axis=0)
        print(f"\n  Total setelah morfologis: {len(windows):,}")

    # ── 3. Fit PCA ────────────────────────────────────────────────────────
    pca = fit_pca(windows, n_components=n_pca)

    # ── 4. Transform ke ruang PCA ─────────────────────────────────────────
    print("\n  Transformasi ke PCA space…")
    reduced = transform_pca(windows, pca)

    # ── 5. SMOTE ──────────────────────────────────────────────────────────
    resampled_r, resampled_l = apply_smote(reduced, labels, target)

    # Hanya simpan synthetic (bukan data asli)
    n_combined = len(resampled_r)
    n_existing = len(reduced)

    if n_combined <= n_existing:
        print("\n  Tidak ada data synthetic yang dihasilkan.")
        return

    syn_reduced = resampled_r[n_existing:]
    syn_labels  = resampled_l[n_existing:]

    # ── 6. Inverse PCA → windows mV ──────────────────────────────────────
    print(f"\n  Inverse PCA untuk {len(syn_reduced):,} synthetic windows…")
    syn_windows = inverse_transform_pca(
        syn_reduced, pca, window_shape=(NUM_CHANNELS, WINDOW_SIZE)
    )

    # Gabungkan dengan morphological (yang tidak masuk SMOTE)
    # Morphological sudah dalam mV; synthetic dari SMOTE inverse-PCA juga mV
    # → simpan gabungan sebagai output

    # ── 7. Simpan output ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: SIMPAN OUTPUT")
    print("=" * 70)

    win_path = output_dir / "synthetic_windows.npy"
    lab_path = output_dir / "synthetic_labels.npy"

    np.save(win_path, syn_windows)
    np.save(lab_path, syn_labels)

    print(f"  ✓ synthetic_windows.npy : {win_path}")
    print(f"    Shape                 : {syn_windows.shape}")
    print(f"    Dtype                 : {syn_windows.dtype}")
    print(f"    Size                  : {win_path.stat().st_size/1024/1024:.1f} MB")
    print(f"  ✓ synthetic_labels.npy  : {lab_path}")

    # Distribusi synthetic
    print("\n  Distribusi synthetic windows:")
    stats = {'n_original': int(n_original), 'n_synthetic': int(len(syn_windows))}
    u, c  = np.unique(syn_labels, return_counts=True)
    for ci, cnt in zip(u, c):
        name = ARRHYTHMIA_CLASSES[int(ci)]
        print(f"    Class {ci:2d} [{name:22s}]: {cnt:,}")
        stats[f'class_{ci}_{name}'] = int(cnt)

    stats_path = output_dir / "smote_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  ✓ smote_stats.json      : {stats_path}")

    print("\n" + "=" * 70)
    print("✓ SMOTE SELESAI")
    print("=" * 70)
    print(f"  Original windows : {n_original:,}")
    print(f"  Synthetic windows: {len(syn_windows):,}")
    print(f"\nGunakan smote_npy_dir='{output_dir}' di HolterECGDataset.")


# ============================================================================
# MAIN / CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SMOTE oversampling untuk dataset ECG Holter"
    )
    parser.add_argument('--labels-csv', type=str,
                        default=str(TRAIN_SPLIT_CSV),
                        help='Path ke train_split.csv')
    parser.add_argument('--data-root', type=str,
                        default=str(HOLTER_FORMAT_DIR),
                        help='Root folder data biner Holter')
    parser.add_argument('--output-dir', type=str,
                        default=str(SMOTE_CACHE_DIR),
                        help='Folder output synthetic windows')
    parser.add_argument('--n-pca', type=int, default=N_PCA_COMPONENTS,
                        help='Jumlah PCA components')
    parser.add_argument('--morph-n', type=int, default=2000,
                        help='Jumlah morphological samples per kelas tanpa data')

    args = parser.parse_args()

    run_smote_pipeline(
        labels_csv=Path(args.labels_csv),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        n_pca=args.n_pca,
        morph_n=args.morph_n,
    )
