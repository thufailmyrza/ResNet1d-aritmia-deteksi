"""
preprocess_ptbxl.py  –  REVISI v3
Preprocessing PTB-XL dataset ke format Holter device.

Perubahan dari v2:
  1. Label: SINGLE-LABEL (integer 0–10) bukan multi-label bitmask.
     Multi-label PTB-XL diselesaikan dengan ARRHYTHMIA_PRIORITY.
  2. Normalisasi: disimpan sebagai int16 (mV × 1000), di-load sebagai mV (/1000).
     Konsisten dengan app yang menerima ecg_signal_mv tanpa normalisasi tambahan.
  3. arrhythmia.bin TIDAK dibuat di preprocess (hanya dibuat saat inference).
     Preprocess hanya menghasilkan labels.csv dengan kolom 'class_label' (int 0–10).
  4. Kolom 'arrhythmia_bitmask' tetap disimpan untuk referensi / analisis.
"""

import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample
import ast
import json
import sys
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config_path import (
    PTBXL_ROOT, PTBXL_DATABASE, HOLTER_FORMAT_DIR,
    ARRHYTHMIA_CLASSES, ARRHYTHMIA_PRIORITY, NUM_ARRHYTHMIA_CLASSES,
    CLASS_TO_IDX,
    HOLTER_SAMPLING_RATE, NUM_CHANNELS, ECG_CHANNELS,
    TRAIN_SPLIT_CSV, VAL_SPLIT_CSV, TEST_SPLIT_CSV,
    LABELS_CSV, STATISTICS_JSON,
    ADC_GAIN,       # = 1000 (mV → int16)
)

LEADS_ORDER     = ECG_CHANNELS
ADC_GAIN_PTBXL  = ADC_GAIN   # PTB-XL sudah dalam mV → ×1000 → int16

# PTBXL → CLASS INDEX MAPPING
# PTB-XL kode SCP → class index (0–10)
# Jika suatu rekaman memiliki beberapa kode, priority diselesaikan di
# labels_to_single_class() menggunakan ARRHYTHMIA_PRIORITY.
PTBXL_TO_CLASS: dict[str, int] = {

    #  1: Premature Beat 
    'PVC':   1,  'VPVC': 1,  'SVPB': 1,
    'PAC':   1,  'SVARR':1,  'EL':   1,
    #  2: Bigeminy 
    'BIGU':  2,
    #  3: Trigeminy 
    'TRIGU': 3,
    #  4: Quadrigeminy  (tidak ada kode eksplisit di PTB-XL) 
    'QUADGU': 4,
    #  5: Couplet  (tidak ada kode eksplisit di PTB-XL) 
    'COUP': 5,
    #  6: Triplet  (tidak ada kode eksplisit di PTB-XL) 
    'TRIP': 6,
    #  7: NSVT (tidak ada kode eksplisit di PTB-XL)
    'NSVT':  7,
    #  8: Tachycardia 
    'STACH': 8,  'SVTAC': 8,  'SVT':  8,
    'PSVT':  8,  'AVNRT': 8,  'AVRT': 8,  'AT': 8,
    #  9: Bradycardia 
    'SBRAD': 9,
    #  10: Atrial Fibrillation 
    'AFIB': 10,  'AF': 10,
    #  Kelas yang DIHAPUS dari mapping baru (tidak dipetakan) 
    # SARRH, AFLT, I-AVB, II-AVB, III-AVB
    # → rekaman dgn hanya kode ini → class 0 (normal)
}

# SIGNAL PROCESSING
def bandpass_filter(signal: np.ndarray, fs: int = HOLTER_SAMPLING_RATE,
                    lowcut: float = 0.5, highcut: float = 40,
                    order: int = 4) -> np.ndarray:
    """Bandpass filter 0.5–40 Hz (menghilangkan baseline wander & HF noise)."""
    nyq  = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)


def resample_signal(signal: np.ndarray, original_fs: int,
                    target_fs: int = HOLTER_SAMPLING_RATE) -> np.ndarray:
    """Resample ke target_fs jika berbeda."""
    if original_fs == target_fs:
        return signal
    num = int(len(signal) * target_fs / original_fs)
    return resample(signal, num, axis=0)


def reorder_leads(signal: np.ndarray, sig_names: list[str]) -> np.ndarray:
    """Susun lead sesuai urutan Holter: I II III aVR aVF aVL V1–V6."""
    result       = np.zeros((signal.shape[0], NUM_CHANNELS), dtype=signal.dtype)
    sig_upper    = [n.upper() for n in sig_names]
    for i, lead in enumerate(LEADS_ORDER):
        if lead.upper() in sig_upper:
            result[:, i] = signal[:, sig_upper.index(lead.upper())]
        else:
            print(f"  Warning: lead {lead} tidak ditemukan → zero-fill")
    return result

# LABEL PROCESSING  –  SINGLE LABEL
def parse_scp_codes(label_str: str) -> list[str]:
    """Parse string dict PTB-XL → list kode SCP."""
    try:
        return list(ast.literal_eval(label_str).keys())
    except Exception:
        return []

def codes_to_class_set(scp_codes: list[str]) -> set[int]:
    """Konversi daftar kode SCP → set class index yang relevan."""
    return {PTBXL_TO_CLASS[c] for c in scp_codes if c in PTBXL_TO_CLASS}

def resolve_single_label(class_set: set[int]) -> int:
    """
    Selesaikan multi-label → single class index menggunakan ARRHYTHMIA_PRIORITY.

    Aturan:
      - Iterasi ARRHYTHMIA_PRIORITY (urutan prioritas tertinggi → terendah).
      - Kembalikan class pertama yang ada di class_set.
      - Jika class_set kosong (tidak ada kode aritmia) → kembalikan 0 (Normal).

    Args:
        class_set: Set class index yang terdeteksi dalam rekaman.

    Returns:
        int: Single class index (0–10).
    """
    if not class_set:
        return 0  # Normal

    for cls in ARRHYTHMIA_PRIORITY:
        if cls in class_set:
            return cls

    return 0  # fallback

def codes_to_bitmask(class_set: set[int]) -> int:
    """Konversi set class index → bitmask (untuk referensi/analisis)."""
    mask = 0
    for cls in class_set:
        mask |= (1 << cls)
    if mask == 0:
        mask = 1  # bit 0 = normal
    return mask

# MAIN CONVERSION FUNCTION
def convert_ptbxl_to_holter_format(save_labels: bool = True,
                                    batch_size: int = 1000):
    """
    Konversi PTB-XL dataset ke format biner Holter (int16, 12-lead).

    Penyimpanan:
      • Signal: float mV × 1000 → int16, disimpan per file (.bin).
      • Label: integer 0–10 (single class) di labels.csv kolom 'class_label'.
      • Bitmask multi-label tetap disimpan di 'arrhythmia_bitmask' untuk referensi.

    Args:
        save_labels:  Simpan labels.csv & statistics.json.
        batch_size:   Jumlah file per sub-folder.

    Returns:
        tuple: (processed_count, label_stats)
    """
    HOLTER_FORMAT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PTB-XL → FORMAT HOLTER  (REVISI v3 – single-label)")
    print("=" * 80)

    if not PTBXL_DATABASE.exists():
        raise FileNotFoundError(f"PTB-XL database tidak ditemukan: {PTBXL_DATABASE}")

    meta = pd.read_csv(PTBXL_DATABASE)
    print(f"Loaded {len(meta)} rekaman")

    # Parse SCP codes → class label
    print("\nParsing kode SCP → single class label…")
    meta['scp_list']    = meta['scp_codes'].apply(parse_scp_codes)
    meta['class_set']   = meta['scp_list'].apply(codes_to_class_set)
    meta['class_label'] = meta['class_set'].apply(resolve_single_label)
    meta['arrhythmia_bitmask'] = meta['class_set'].apply(codes_to_bitmask)
    meta['has_arrhythmia']     = (meta['class_label'] != 0).astype(int)

    total = len(meta)

    # Distribusi kelas
    print("\n" + "=" * 80)
    print("DISTRIBUSI KELAS (single-label)")
    print("=" * 80)
    for cls_idx in range(NUM_ARRHYTHMIA_CLASSES):
        cnt  = (meta['class_label'] == cls_idx).sum()
        name = ARRHYTHMIA_CLASSES[cls_idx]
        flag = 1 << cls_idx
        print(f"  Class {cls_idx:2d} [{name:22s}] flag=2^{cls_idx:<2d}={flag:<5d} : "
              f"{cnt:6,}  ({cnt/total*100:5.2f}%)")

    # Verifikasi lead
    print("\n" + "=" * 80)
    print("VERIFIKASI LEAD")
    print("=" * 80)
    try:
        _, si = wfdb.rdsamp(str(PTBXL_ROOT / meta.iloc[0]['filename_hr']))
        su    = [n.upper() for n in si['sig_name']]
        miss  = [l for l in LEADS_ORDER if l.upper() not in su]
        print(f"  PTB-XL    : {si['sig_name']}")
        print(f"  Expected  : {LEADS_ORDER}")
        print(f"  {'✓ Semua ada' if not miss else f'⚠ Hilang: {miss}'}")
    except Exception as e:
        print(f"  ⚠ Tidak bisa verifikasi: {e}")

    # Konversi rekaman
    print("\n" + "=" * 80)
    print(f"KONVERSI {total:,} REKAMAN")
    print("=" * 80)

    all_labels    = []
    ok_count      = 0
    err_count     = 0
    n_samples_log = []

    pbar = tqdm(meta.iterrows(), total=total, desc="Konversi")

    for _, row in pbar:
        try:
            record_path       = PTBXL_ROOT / row['filename_hr']
            signal, info      = wfdb.rdsamp(str(record_path))

            if info['fs'] != HOLTER_SAMPLING_RATE:
                signal = resample_signal(signal, info['fs'])

            reordered  = reorder_leads(signal, info['sig_name'])
            filtered   = bandpass_filter(reordered)

            # PTB-XL sudah dalam mV → ×1000 → int16
            int16_sig  = np.clip(filtered * ADC_GAIN_PTBXL,
                                 -32768, 32767).astype(np.int16)

            folder_idx = ok_count // batch_size
            folder     = HOLTER_FORMAT_DIR / f"batch_{folder_idx:05d}"
            folder.mkdir(exist_ok=True)

            out_fname  = f"{Path(row['filename_hr']).stem}.bin"
            int16_sig.tofile(folder / out_fname)

            n_samp = len(int16_sig)
            n_samples_log.append(n_samp)

            all_labels.append({
                'record_id':          ok_count,
                'original_filename':  row['filename_hr'],
                'output_filename':    out_fname,
                'batch_dir':          folder.name,
                'class_label':        int(row['class_label']),      # ← SINGLE LABEL
                'arrhythmia_bitmask': int(row['arrhythmia_bitmask']),
                'has_arrhythmia':     int(row['has_arrhythmia']),
                'n_samples':          n_samp,
                'scp_codes':          row['scp_codes'],
                'age':  int(row['age']) if pd.notna(row['age']) else -1,
                'sex':  int(row['sex']) if pd.notna(row['sex']) else -1,
                'report': row['report'] if pd.notna(row['report']) else '',
                'success': True,
            })
            ok_count += 1
            pbar.set_postfix({'ok': ok_count, 'err': err_count})

        except Exception as e:
            err_count += 1
            all_labels.append({
                'record_id': len(all_labels),
                'original_filename': row.get('filename_hr', '?'),
                'output_filename': '', 'batch_dir': '',
                'class_label': 0, 'arrhythmia_bitmask': 0,
                'has_arrhythmia': 0, 'n_samples': 0,
                'scp_codes': '', 'age': -1, 'sex': -1,
                'report': f"Error: {e}", 'success': False,
            })
            pbar.set_postfix({'ok': ok_count, 'err': err_count})

    # Ringkasan
    print(f"\n✓ Berhasil : {ok_count:,}")
    print(f"✗ Gagal    : {err_count:,}")
    print(f"  Sukses   : {ok_count/(ok_count+err_count)*100:.2f}%")

    label_stats = {}

    if save_labels and all_labels:
        labels_df   = pd.DataFrame(all_labels)
        labels_df.to_csv(LABELS_CSV, index=False)
        print(f"\n✓ labels.csv         : {LABELS_CSV}")

        # Statistik per kelas
        success_df  = labels_df[labels_df['success']].copy()
        label_stats = {
            'total':           len(labels_df),
            'success':         len(success_df),
            'failed':          len(labels_df) - len(success_df),
            'has_arrhythmia':  int(success_df['has_arrhythmia'].sum()),
            'normal':          int((success_df['class_label'] == 0).sum()),
        }
        for idx, name in ARRHYTHMIA_CLASSES.items():
            cnt = int((success_df['class_label'] == idx).sum())
            label_stats[f'class_{idx}_{name}'] = cnt

        with open(STATISTICS_JSON, 'w') as f:
            json.dump(label_stats, f, indent=2)
        print(f"✓ statistics.json    : {STATISTICS_JSON}")

    return ok_count, label_stats

# TRAIN / VAL / TEST SPLIT
def create_train_val_test_splits(train_ratio: float = 0.80,
                                  val_ratio:   float = 0.10,
                                  test_ratio:  float = 0.10,
                                  random_seed: int   = 42):
    """
    Bagi dataset menjadi train/val/test dengan stratifikasi per class_label.

    Stratifikasi berdasarkan class_label (bukan has_arrhythmia) agar distribusi
    kelas terjaga di setiap split.
    """
    print("=" * 80)
    print("MEMBUAT TRAIN / VAL / TEST SPLITS")
    print("=" * 80)

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Jumlah rasio harus = 1.0")

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV tidak ditemukan: {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)
    df = df[df['success'] == True].reset_index(drop=True)

    print(f"Total rekaman : {len(df):,}")

    # Kelas dengan jumlah sangat sedikit tidak bisa di-stratify secara langsung
    # → gunakan has_arrhythmia sebagai stratify proxy
    stratify_col = df['class_label']
    min_per_class = stratify_col.value_counts().min()

    if min_per_class < 4:
        print("  ⚠ Beberapa kelas terlalu sedikit untuk stratifikasi per kelas.")
        print("    Menggunakan has_arrhythmia sebagai proxy stratifikasi.")
        stratify_col = df['has_arrhythmia']

    train_df, val_test_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        random_state=random_seed, stratify=stratify_col
    )
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_seed,
        stratify=val_test_df['has_arrhythmia']
    )

    for name, sdf in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n  {name}: {len(sdf):,}")
        for cls, cname in ARRHYTHMIA_CLASSES.items():
            cnt = (sdf['class_label'] == cls).sum()
            if cnt > 0:
                print(f"    Class {cls:2d} [{cname:22s}]: {cnt:5,}")

    train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(VAL_SPLIT_CSV,     index=False)
    test_df.to_csv(TEST_SPLIT_CSV,   index=False)
    print(f"\n✓ Splits disimpan di: {HOLTER_FORMAT_DIR}")

    return {'train': train_df, 'val': val_df, 'test': test_df}

# MAIN
if __name__ == "__main__":
    print("=" * 80)
    print("PREPROCESSING PTB-XL → HOLTER  (v3 single-label, mV scale)")
    print("=" * 80)
    print(f"  Sampling rate    : {HOLTER_SAMPLING_RATE} Hz")
    print(f"  Channels         : {NUM_CHANNELS}")
    print(f"  Num classes      : {NUM_ARRHYTHMIA_CLASSES}")
    print(f"  ADC gain PTB-XL  : {ADC_GAIN_PTBXL} (mV → int16)")
    print(f"  Load norm        : / {ADC_GAIN_PTBXL}  (int16 → mV, cocok dengan app)")

    try:
        ok, stats = convert_ptbxl_to_holter_format(save_labels=True, batch_size=1000)
        print(f"\n✓ Konversi selesai: {ok:,} rekaman")
    except Exception as e:
        import traceback; traceback.print_exc(); exit(1)

    try:
        create_train_val_test_splits(
            train_ratio=0.80, val_ratio=0.10, test_ratio=0.10)
        print("\n✓ Split selesai")
    except Exception as e:
        import traceback; traceback.print_exc(); exit(1)

    print("\n" + "=" * 80)
    print("LANGKAH BERIKUTNYA:")
    print("  1. python smote_oversampling.py       ← balance kelas minoritas")
    print("  2. python train_model.py              ← training")
    print("  3. python inference_export_pkl.py export ← export ONNX + PKL")
    print("=" * 80)
