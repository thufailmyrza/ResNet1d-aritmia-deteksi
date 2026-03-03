"""
merge_datasets.py
=================
Gabungkan dataset latih PTB-XL + INCART menjadi satu dataset unified.

Kontrak:
  • Membaca  : LABELS_CSV (PTB-XL)  →  resolve arrhythmia_bitmask → class_index
               INCART_LABELS_CSV     →  class_index sudah tersedia langsung
  • Menulis  : MERGED_LABELS_CSV    (semua baris gabungan)
               MERGED_STATS_JSON
               TRAIN_SPLIT_CSV  ─┐
               VAL_SPLIT_CSV     ├─ MENIMPA file di HOLTER_FORMAT_DIR
               TEST_SPLIT_CSV   ─┘   sehingga training code tidak berubah

Resolusi PTB-XL bitmask → class_index:
  PTB-XL menyimpan 'arrhythmia_bitmask' (bit per kelas lama).
  Kita map bit lama → set nama kelas baru → resolve via ARRHYTHMIA_PRIORITY.
  Mapping lama→baru ada di PTBXL_BIT_TO_CLASS di bawah.

Stratified split:
  3 strata: (0) Normal, (1) Aritmia "lama" (ada di PTB-XL),
            (2) Aritmia "baru" (Quadrigeminy/Couplet/Triplet/NSVT)
  Ini memastikan kelas baru dari INCART terwakili di train/val/test.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_path import (
    LABELS_CSV, TRAIN_SPLIT_CSV, VAL_SPLIT_CSV, TEST_SPLIT_CSV,
    INCART_LABELS_CSV, MERGED_LABELS_CSV, MERGED_STATS_JSON,
    ARRHYTHMIA_CLASSES, CLASS_TO_IDX, ARRHYTHMIA_PRIORITY,
    NUM_ARRHYTHMIA_CLASSES, HOLTER_FORMAT_DIR,
)

# ============================================================================
# MAPPING: PTB-XL arrhythmia_bitmask  →  nama kelas baru
# ============================================================================
# PTB-XL preprocess_ptbxl.py menggunakan ARRHYTHMIA_BIT_MAPPING lama
# (sebelum v3) yang mungkin berbeda urutan bit-nya.
# Kita petakan bit lama → nama kelas baru yang ada di ARRHYTHMIA_CLASSES v3.
#
# Bit lama (dari preprocess_ptbxl.py versi sebelum v3):
#   0 = tachycardia, 1 = bradycardia, 2 = sinus_arrhythmia,
#   3 = pvc_bigeminy, 4 = pvc_trigeminy, 5 = premature_beat,
#   6 = atrial_fibrillation, 7 = atrial_flutter,
#   8..11 = av_block / svt
#
# Petakan ke nama kelas v3 yang tersedia:
# (Nama lain yang tidak ada di v3 di-drop → akan fallback ke normal
#  atau kelas lain yang prioritasnya lebih tinggi)
# ============================================================================

# bit_posisi_lama → nama kelas v3  (None = tidak dipetakan ke kelas manapun)
PTBXL_OLD_BIT_TO_CLASS = {
    0:  'tachycardia',
    1:  'bradycardia',
    2:  None,                    # sinus_arrhythmia tidak ada di v3
    3:  'bigeminy',              # pvc_bigeminy
    4:  'trigeminy',             # pvc_trigeminy
    5:  'premature_beat',
    6:  'atrial_fibrillation',
    7:  'atrial_fibrillation',   # atrial_flutter → AF (paling dekat)
    8:  None,                    # 1st degree AV block tidak ada di v3
    9:  None,                    # 2nd degree AV block
    10: None,                    # 3rd degree AV block
    11: 'tachycardia',           # SVT → tachycardia
}

# ============================================================================
# KONSTANTA SPLIT
# ============================================================================

TRAIN_RATIO  = 0.75
VAL_RATIO    = 0.15
TEST_RATIO   = 0.10
RANDOM_SEED  = 42

# Kelas yang HANYA datang dari INCART (baru di v3)
NEW_CLASSES  = {'quadrigeminy', 'couplet', 'triplet', 'nsvt'}


# ============================================================================
# LOAD PTB-XL LABELS
# ============================================================================

def _bitmask_to_class_set(bitmask: int) -> set:
    """
    Konversi arrhythmia_bitmask PTB-XL (format lama) → set nama kelas v3.
    """
    classes = set()
    for bit, name in PTBXL_OLD_BIT_TO_CLASS.items():
        if name and (bitmask & (1 << bit)):
            classes.add(name)
    return classes


def _resolve_class(class_set: set) -> int:
    """Resolve set kelas → satu class_index via ARRHYTHMIA_PRIORITY."""
    for idx in ARRHYTHMIA_PRIORITY:
        if ARRHYTHMIA_CLASSES[idx] in class_set:
            return idx
    return CLASS_TO_IDX['normal']


def load_ptbxl() -> pd.DataFrame:
    """Load PTB-XL labels.csv dan resolve bitmask → class_index."""
    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"PTB-XL labels tidak ditemukan: {LABELS_CSV}\n"
            "Jalankan preprocess_ptbxl.py terlebih dahulu."
        )

    df = pd.read_csv(LABELS_CSV)
    df = df[df['success'] == True].copy()
    df['source'] = 'ptbxl'

    # Deteksi kolom bitmask (backward compat: 'arrhythmia_bitmask' atau 'arrhythmia_mask')
    if 'arrhythmia_bitmask' in df.columns:
        bitmask_col = 'arrhythmia_bitmask'
    elif 'arrhythmia_mask' in df.columns:
        bitmask_col = 'arrhythmia_mask'
    else:
        raise KeyError("Kolom 'arrhythmia_bitmask' tidak ditemukan di PTB-XL labels.csv")

    # Resolve bitmask → class_index
    df['class_index'] = df[bitmask_col].apply(
        lambda bm: _resolve_class(_bitmask_to_class_set(int(bm)))
    )
    df['class_name']     = df['class_index'].map(ARRHYTHMIA_CLASSES)
    df['has_arrhythmia'] = (df['class_index'] > 0).astype(int)

    # Normalkan kolom filepath
    if 'filepath' not in df.columns:
        # Rekonstruksi dari kolom batch_dir + output_filename jika ada
        if 'batch_dir' in df.columns and 'output_filename' in df.columns:
            from config_path import HOLTER_FORMAT_DIR as HFD
            df['filepath'] = df.apply(
                lambda r: str(HFD / r['batch_dir'] / r['output_filename']), axis=1
            )
        else:
            df['filepath'] = ''

    print(f"✓ PTB-XL loaded: {len(df):,} windows")
    _print_class_dist(df, "PTB-XL (setelah resolve)")
    return df


# ============================================================================
# LOAD INCART LABELS
# ============================================================================

def load_incart() -> pd.DataFrame:
    """Load INCART labels (sudah class_index, tidak perlu resolve)."""
    if not INCART_LABELS_CSV.exists():
        raise FileNotFoundError(
            f"INCART labels tidak ditemukan: {INCART_LABELS_CSV}\n"
            "Jalankan preprocess_incart.py terlebih dahulu."
        )

    df = pd.read_csv(INCART_LABELS_CSV)
    df = df[df['success'] == True].copy()
    df['source'] = 'incart'

    # Pastikan kolom standar ada
    df['class_name']     = df['class_index'].map(ARRHYTHMIA_CLASSES)
    df['has_arrhythmia'] = (df['class_index'] > 0).astype(int)

    # ── NORMALISASI KOLOM ke format holter_dataset.py ─────────────────────────
    # holter_dataset._build_window_index() membaca: batch_dir, output_filename, class_label
    # INCART CSV punya: filepath, class_index
    # → Derive batch_dir + output_filename dari filepath
    # → Tambah class_label sebagai alias class_index

    if 'filepath' in df.columns:
        def _split_path(fp):
            """Pecah filepath absolut → (batch_dir, output_filename)."""
            try:
                p = Path(str(fp))
                return p.parent.name, p.name   # e.g. ("batch_00000", "incart_000000.bin")
            except Exception:
                return None, None

        splits = df['filepath'].apply(_split_path)
        df['batch_dir']       = [s[0] for s in splits]
        df['output_filename'] = [s[1] for s in splits]

        # data_root untuk INCART = parent dari batch_XXXXX = INCART_FORMAT_DIR
        # holter_dataset akan diberi data_root=HOLTER_FORMAT_DIR untuk PTB-XL
        # tapi untuk INCART kita simpan filepath absolut sebagai fallback
        # → holter_dataset._resolve_bin_path() akan pakai filepath jika batch_dir gagal

    # class_label = alias untuk class_index (agar holter_dataset bisa baca)
    df['class_label'] = df['class_index']

    print(f"✓ INCART loaded : {len(df):,} windows")
    _print_class_dist(df, "INCART")
    return df


# ============================================================================
# MERGE
# ============================================================================

def merge(ptbxl: pd.DataFrame, incart: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan kedua DataFrame, align kolom, return merged DataFrame."""
    all_cols = sorted(set(ptbxl.columns) | set(incart.columns))
    ptbxl    = ptbxl.reindex(columns=all_cols)
    incart   = incart.reindex(columns=all_cols)
    merged   = pd.concat([ptbxl, incart], ignore_index=True)

    # ── Pastikan kolom kritis tidak NaN ──────────────────────────────────────
    # class_index dan class_label harus selalu ada dan sinkron
    if 'class_index' in merged.columns and 'class_label' in merged.columns:
        # Isi NaN class_index dari class_label dan sebaliknya
        merged['class_index'] = merged['class_index'].fillna(merged['class_label']).fillna(0).astype(int)
        merged['class_label'] = merged['class_label'].fillna(merged['class_index']).fillna(0).astype(int)
    elif 'class_label' in merged.columns:
        merged['class_index'] = merged['class_label'].fillna(0).astype(int)
    elif 'class_index' in merged.columns:
        merged['class_label'] = merged['class_index'].fillna(0).astype(int)

    merged['has_arrhythmia'] = (merged['class_index'] > 0).astype(int)
    merged['class_name']     = merged['class_index'].map(ARRHYTHMIA_CLASSES)

    # ── Verifikasi tidak ada NaN di kolom path kritis ─────────────────────────
    # PTB-XL rows: batch_dir + output_filename valid, filepath mungkin NaN → OK
    # INCART rows: filepath valid, batch_dir + output_filename di-derive → OK
    n_no_path = merged[
        merged['filepath'].isna() &
        (merged['batch_dir'].isna() | merged['output_filename'].isna())
    ].shape[0]
    if n_no_path > 0:
        print(f"  ⚠ {n_no_path} baris tidak punya path yang valid (akan dilewati saat training)")

    return merged


# ============================================================================
# HELPER
# ============================================================================

def _print_class_dist(df: pd.DataFrame, title: str):
    n = len(df)
    print(f"\n  [{title}]  total={n:,}")
    for idx in range(NUM_ARRHYTHMIA_CLASSES):
        count = int((df['class_index'] == idx).sum())
        if count > 0:
            name = ARRHYTHMIA_CLASSES[idx]
            marker = " ← BARU" if name in NEW_CLASSES else ""
            print(f"    [{idx:2d}] {name:25s}: {count:6,} ({count/n*100:5.1f}%){marker}")


def _strat_key(class_index: int) -> int:
    """
    Stratification key:
      0 = Normal
      1 = Aritmia kelas lama (ada di PTB-XL)
      2 = Aritmia kelas baru (Quadrigeminy/Couplet/Triplet/NSVT – dari INCART)
    """
    if class_index == 0:
        return 0
    name = ARRHYTHMIA_CLASSES.get(class_index, '')
    return 2 if name in NEW_CLASSES else 1


# ============================================================================
# SPLIT
# ============================================================================

def create_splits(merged: pd.DataFrame) -> dict:
    """Buat train/val/test split dengan stratifikasi 3 strata."""
    print("\n" + "=" * 60)
    print("TRAIN / VAL / TEST SPLIT")
    print("=" * 60)

    merged = merged.copy()
    merged['_strat'] = merged['class_index'].apply(_strat_key)

    # Buang strata dengan < 3 sampel (tidak cukup untuk split)
    valid = merged['_strat'].value_counts()
    valid = valid[valid >= 3].index
    dropped = len(merged) - len(merged[merged['_strat'].isin(valid)])
    if dropped:
        print(f"  ⚠ {dropped} baris dengan strata < 3 sampel dibuang")
    merged = merged[merged['_strat'].isin(valid)].copy()

    for k, lbl in {0: 'Normal', 1: 'Aritmia lama', 2: 'Aritmia baru'}.items():
        n = (merged['_strat'] == k).sum()
        print(f"  Strata {k} ({lbl}): {n:,}")

    # Split 1: train vs (val+test)
    train, val_test = train_test_split(
        merged,
        test_size    = VAL_RATIO + TEST_RATIO,
        random_state = RANDOM_SEED,
        stratify     = merged['_strat'],
    )

    # Split 2: val vs test
    val, test = train_test_split(
        val_test,
        test_size    = TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state = RANDOM_SEED,
        stratify     = val_test['_strat'],
    )

    for split in (train, val, test):
        split.drop(columns=['_strat'], inplace=True, errors='ignore')

    n_total = len(merged)
    print(f"\n  Train  : {len(train):7,} ({len(train)/n_total*100:.1f}%)")
    print(f"  Val    : {len(val):7,} ({len(val)/n_total*100:.1f}%)")
    print(f"  Test   : {len(test):7,} ({len(test)/n_total*100:.1f}%)")

    return {'train': train, 'val': val, 'test': test}


# ============================================================================
# SAVE
# ============================================================================

def save(merged: pd.DataFrame, splits: dict) -> dict:
    """Simpan semua output ke disk."""
    HOLTER_FORMAT_DIR.mkdir(parents=True, exist_ok=True)

    merged.to_csv(MERGED_LABELS_CSV, index=False)
    splits['train'].to_csv(TRAIN_SPLIT_CSV, index=False)
    splits['val'].to_csv(VAL_SPLIT_CSV,   index=False)
    splits['test'].to_csv(TEST_SPLIT_CSV,  index=False)

    n = len(merged)
    stats = {
        'total_windows':     n,
        'ptbxl_windows':     int((merged['source'] == 'ptbxl').sum()),
        'incart_windows':    int((merged['source'] == 'incart').sum()),
        'train_size':        len(splits['train']),
        'val_size':          len(splits['val']),
        'test_size':         len(splits['test']),
        'total_arrhythmia':  int(merged['has_arrhythmia'].sum()),
        'total_normal':      int((merged['class_index'] == 0).sum()),
        'per_class':         {},
    }

    for idx in range(NUM_ARRHYTHMIA_CLASSES):
        name  = ARRHYTHMIA_CLASSES[idx]
        count = int((merged['class_index'] == idx).sum())
        stats['per_class'][name] = {
            'class_index': idx,
            'count':       count,
            'pct':         round(count / n * 100, 2) if n else 0,
            'source':      'incart_only' if name in NEW_CLASSES else 'both',
        }

    with open(MERGED_STATS_JSON, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ {MERGED_LABELS_CSV}")
    print(f"✓ {MERGED_STATS_JSON}")
    print(f"✓ {TRAIN_SPLIT_CSV}   ← training code membaca ini (tidak berubah)")
    print(f"✓ {VAL_SPLIT_CSV}")
    print(f"✓ {TEST_SPLIT_CSV}")

    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("MERGE DATASET: PTB-XL + ST. PETERSBURG INCART  (v3.1 single-label)")
    print("=" * 80)

    ptbxl_df  = load_ptbxl()
    incart_df = load_incart()

    merged = merge(ptbxl_df, incart_df)

    print("\n" + "=" * 60)
    _print_class_dist(merged, "MERGED (total)")
    print("=" * 60)

    splits = create_splits(merged)
    stats  = save(merged, splits)

    print("\n" + "=" * 80)
    print("✅ MERGE SELESAI")
    print("=" * 80)
    print(f"  Total  : {stats['total_windows']:,} windows")
    print(f"  PTB-XL : {stats['ptbxl_windows']:,}")
    print(f"  INCART : {stats['incart_windows']:,}")
    print(f"\n  Kelas baru (hanya dari INCART):")
    for name in ('quadrigeminy', 'couplet', 'triplet', 'nsvt'):
        print(f"    {name:20s}: {stats['per_class'][name]['count']:,}")
    print(f"\n  Jalankan training seperti biasa – split CSV sudah diperbarui.")


if __name__ == "__main__":
    main()