"""
preprocess_incart.py  –  v5  (ground-truth driven)
====================================================
Konversi St. Petersburg INCART 12-Lead Arrhythmia Database
ke format training identik dengan output preprocess_ptbxl.py.

Perubahan UTAMA v5:
  Label kelas ditentukan dari record-descriptions.txt (ground truth),
  BUKAN dari deteksi algoritmik beat-level.

  Arsitektur labeling dua tingkat:
    ┌──────────────────────────────────────────────────────────────┐
    │  Level 1 – RECORD LEVEL  (dari record-descriptions.txt)     │
    │  "ventricular couplets" → primary_class = couplet (5)       │
    │  Menentukan JENIS aritmia yang bisa muncul di rekaman ini    │
    ├──────────────────────────────────────────────────────────────┤
    │  Level 2 – WINDOW LEVEL  (dari beat annotations .atr)       │
    │  Apakah window ini punya aktivitas aritmia?                  │
    │  YES → gunakan primary_class rekaman                         │
    │  NO  → normal (class 0)                                      │
    └──────────────────────────────────────────────────────────────┘

  Window dianggap "aktif" (ada aritmia) jika:
    • Kelas = VEB-pattern (premature/bigeminy/trigeminy/quadrigeminy/
                           couplet/triplet/nsvt):
        ada ≥1 ectopic beat (VEB atau SVEB) dalam window
    • Kelas = tachycardia:
        HR (dari semua beat) > 100 bpm
    • Kelas = bradycardia:
        HR (dari semua beat) < 60 bpm, ≥2 beat dalam window
    • Kelas = atrial_fibrillation:
        rr_cv > 0.20 dan veb_frac < 0.20
    • Kelas = normal (I60):
        selalu normal

Kontrak output (cocok dengan kontrak model v3):
  • Binary file  : (2500, 12) int16, row-major  [sample × channel]
  • Labels CSV   : kolom utama = 'class_index' (0–10)
  • Normalisasi  : int16 / ADC_GAIN_INT16 = mV
"""

import sys
import json
import re
import argparse
import traceback
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_path import (
    INCART_ROOT, INCART_FORMAT_DIR, INCART_LABELS_CSV, INCART_STATS_JSON,
    HOLTER_SAMPLING_RATE, NUM_CHANNELS, ECG_CHANNELS,
    ADC_GAIN_INT16, WINDOW_SIZE,
    ARRHYTHMIA_CLASSES, CLASS_TO_IDX, ARRHYTHMIA_PRIORITY,
    NUM_ARRHYTHMIA_CLASSES,
)

# ============================================================================
# KONSTANTA
# ============================================================================

INCART_FS      = 257
TARGET_FS      = HOLTER_SAMPLING_RATE   # 500
WINDOW_SAMPLES = WINDOW_SIZE            # 2500
STRIDE_SAMPLES = 500
BATCH_SIZE     = 1000

_VEB          = {'V', 'E', 'F'}
_SVEB         = {'S', 'A', 'a', 'J', 'j', 'e'}
_ALL_ECTOPIC  = _VEB | _SVEB
_BEAT_SYMBOLS = set('NVSAaJjEeFf')

_HR_TACHY     = 100    # bpm
_HR_BRADY     = 60     # bpm
_AF_CV_THR    = 0.20   # RR coefficient of variation untuk AF
_AF_VEB_MAX   = 0.20   # max VEB fraction untuk AF (lebih longgar dari v4)


# ============================================================================
# LEVEL 1: PARSE RECORD-DESCRIPTIONS.TXT  →  PER-RECORD PRIMARY CLASS
# ============================================================================

# Keyword matching order: lebih spesifik dulu
_KEYWORD_MAP = [
    (r'paroxysmal\s+vt',               'nsvt'),
    (r'paroxysmal\s+ventricular',       'nsvt'),
    (r'ventricular\s+tachycardia',      'nsvt'),
    (r'\bvt\b',                         'nsvt'),
    (r'ventricular\s+rhythm',           'nsvt'),
    (r'atrial\s+fibrillation',          'atrial_fibrillation'),
    (r'triplet',                        'triplet'),
    (r'trigeminy',                      'trigeminy'),
    (r'bigeminy',                       'bigeminy'),
    (r'couplet',                        'couplet'),
    (r'quadrigeminy',                   'quadrigeminy'),
    (r'\btachycardia\b',                'tachycardia'),
    (r'\bbradycardia\b',                'bradycardia'),
    (r'\bpvcs?\b',                      'premature_beat'),
    (r'\bsveb',                         'premature_beat'),
    (r'\bapc',                          'premature_beat'),
    (r'escape\s+beat',                  'premature_beat'),
    (r'fusion\s+beat',                  'premature_beat'),
]


def parse_record_descriptions(path: Path) -> dict:
    """
    Parse record-descriptions.txt menjadi dict:
        { 'I01': set_of_class_names, 'I02': set_of_class_names, ... }

    Format file:
        I01
        PVCs, noise
        I02
        ventricular trigeminy, ventricular couplets
        ...
    """
    if not path.exists():
        return {}

    text      = path.read_text(encoding='utf-8', errors='replace')
    lines     = [l.strip() for l in text.splitlines()]
    records   = {}
    current   = None

    for line in lines:
        if re.match(r'^I\d{2}$', line):
            current = line
            records[current] = set()
        elif current and line:
            desc_lower = line.lower()
            for pattern, cls in _KEYWORD_MAP:
                if re.search(pattern, desc_lower):
                    records[current].add(cls)

    return records


def resolve_primary_class(class_set: set) -> int:
    """Resolve set kelas → satu class_index via ARRHYTHMIA_PRIORITY."""
    for idx in ARRHYTHMIA_PRIORITY:
        if ARRHYTHMIA_CLASSES[idx] in class_set:
            return idx
    return CLASS_TO_IDX['normal']


def build_record_class_map(descriptions_path: Path) -> dict:
    """
    Return dict: { 'I01': primary_class_index, 'I02': ..., ... }
    Sekaligus print summary.
    """
    raw   = parse_record_descriptions(descriptions_path)
    result = {}

    print("\n  Parsing record-descriptions.txt:")
    for rec in sorted(raw.keys()):
        cls_set = raw[rec]
        primary = resolve_primary_class(cls_set)
        result[rec] = primary
        name    = ARRHYTHMIA_CLASSES[primary]
        new_m   = " ← BARU" if name in ('quadrigeminy','couplet','triplet','nsvt') else ""
        print(f"    {rec}: [{primary:2d}] {name:20s}  {sorted(cls_set)}{new_m}")

    return result


# ============================================================================
# LEVEL 2: WINDOW ACTIVITY DETECTION  (beat annotations)
# ============================================================================

def _window_is_active(primary_class: int,
                       beat_smp: np.ndarray,
                       beat_sym: np.ndarray,
                       win_start: int,
                       win_end: int) -> bool:
    """
    Tentukan apakah window ini mengandung aritmia sesuai primary_class rekaman.

    Returns:
        True  → window aktif  → label = primary_class
        False → window normal → label = 0
    """
    mask  = (beat_smp >= win_start) & (beat_smp < win_end)
    w_smp = beat_smp[mask]
    w_sym = beat_sym[mask]
    n     = len(w_smp)
    name  = ARRHYTHMIA_CLASSES.get(primary_class, 'normal')

    # Normal record (I60) → selalu False
    if primary_class == CLASS_TO_IDX['normal']:
        return False

    # VEB-pattern arrhythmias: aktif jika ada ectopic beat apapun
    if name in ('premature_beat', 'bigeminy', 'trigeminy', 'quadrigeminy',
                'couplet', 'triplet', 'nsvt'):
        return any(s in _ALL_ECTOPIC for s in w_sym)

    # Tachycardia: aktif jika HR > 100 bpm (dari semua beat)
    if name == 'tachycardia':
        if n >= 2:
            rr_sec = np.diff(w_smp.astype(float)) / TARGET_FS
            hr = 60.0 / np.mean(rr_sec) if np.mean(rr_sec) > 0 else 0
            return hr >= _HR_TACHY
        return False

    # Bradycardia: aktif jika HR < 60 bpm
    if name == 'bradycardia':
        if n >= 2:
            rr_sec = np.diff(w_smp.astype(float)) / TARGET_FS
            hr = 60.0 / np.mean(rr_sec) if np.mean(rr_sec) > 0 else 0
            return 0 < hr <= _HR_BRADY
        return False

    # Atrial Fibrillation: aktif jika RR sangat irregular
    if name == 'atrial_fibrillation':
        if n >= 4:
            is_veb   = np.array([s in _VEB for s in w_sym])
            veb_frac = is_veb.mean()
            rr_sec   = np.diff(w_smp.astype(float)) / TARGET_FS
            rr_cv    = np.std(rr_sec) / np.mean(rr_sec) if np.mean(rr_sec) > 0 else 0
            # AF: irregular DAN tidak dominan VEB
            return rr_cv > _AF_CV_THR and veb_frac < _AF_VEB_MAX
        return False

    # Default: aktif jika ada ectopic beat
    return any(s in _ALL_ECTOPIC for s in w_sym)


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def _norm_lead(name: str) -> str:
    return name.strip().upper()


def _reorder_leads(signal: np.ndarray, src_names: list) -> np.ndarray:
    src_norm = [_norm_lead(n) for n in src_names]
    tgt_norm = [_norm_lead(n) for n in ECG_CHANNELS]
    out      = np.zeros((signal.shape[0], NUM_CHANNELS), dtype=signal.dtype)
    for i, tgt in enumerate(tgt_norm):
        if tgt in src_norm:
            out[:, i] = signal[:, src_norm.index(tgt)]
    return out


def _bandpass(signal: np.ndarray, fs: float) -> np.ndarray:
    nyq  = fs / 2.0
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
    out  = np.empty_like(signal, dtype=np.float64)
    for ch in range(signal.shape[1]):
        try:
            out[:, ch] = filtfilt(b, a, signal[:, ch].astype(np.float64))
        except Exception:
            out[:, ch] = signal[:, ch].astype(np.float64)
    return out


def _resample_signal(signal: np.ndarray, orig_fs: float, tgt_fs: float) -> np.ndarray:
    if abs(orig_fs - tgt_fs) < 0.5:
        return signal.astype(np.float64)
    n_tgt = int(round(signal.shape[0] * tgt_fs / orig_fs))
    out   = np.empty((n_tgt, signal.shape[1]), dtype=np.float64)
    for ch in range(signal.shape[1]):
        out[:, ch] = resample(signal[:, ch].astype(np.float64), n_tgt)
    return out


def _rescale_ann(samples, orig_fs: float, tgt_fs: float) -> np.ndarray:
    return np.round(np.asarray(samples, dtype=np.float64) * tgt_fs / orig_fs).astype(np.int64)


# ============================================================================
# PROSES SATU REKAMAN
# ============================================================================

def _process_record(rec_name: str,
                    rec_dir: Path,
                    primary_class: int,
                    out_dir: Path,
                    start_idx: int,
                    verbose: bool = False) -> tuple:
    """
    Konversi satu rekaman INCART → (list_of_dicts, error_or_None).

    Args:
        primary_class : class_index dari record-descriptions.txt
    """
    labels   = []
    file_idx = start_idx
    rec_path = rec_dir / rec_name

    try:
        # ── Signal (sudah mV dari rdsamp) ────────────────────────────────────
        signal, info = wfdb.rdsamp(str(rec_path))
        orig_fs   = float(info['fs'])
        sig_names = [str(n) for n in info['sig_name']]

        signal = _reorder_leads(signal, sig_names)
        signal = _resample_signal(signal, orig_fs, TARGET_FS)
        signal = _bandpass(signal, TARGET_FS)

        # ── Beat annotations ─────────────────────────────────────────────────
        ann       = wfdb.rdann(str(rec_path), 'atr')
        beat_mask = np.array([str(s) in _BEAT_SYMBOLS for s in ann.symbol])
        beat_smp  = _rescale_ann(ann.sample[beat_mask], orig_fs, TARGET_FS)
        beat_sym  = np.array([str(s) for s in ann.symbol])[beat_mask]

        if verbose:
            from collections import Counter
            cnt = Counter(beat_sym.tolist())
            print(f"\n  [{rec_name}] primary_class=[{primary_class}] "
                  f"{ARRHYTHMIA_CLASSES[primary_class]}")
            print(f"  [{rec_name}] beat counts: {dict(cnt)}")
            print(f"  [{rec_name}] signal: {signal.shape[0]} samples = "
                  f"{signal.shape[0]/TARGET_FS:.0f}s @ {TARGET_FS}Hz")

        n_total   = signal.shape[0]
        win_start = 0
        n_active  = 0
        n_normal  = 0

        while win_start + WINDOW_SAMPLES <= n_total:
            win_end = win_start + WINDOW_SAMPLES
            window  = signal[win_start:win_end, :]

            # mV × 1000 → int16
            win_int16 = np.clip(
                window * ADC_GAIN_INT16, -32768, 32767
            ).astype(np.int16)

            folder_idx = file_idx // BATCH_SIZE
            folder     = out_dir / f"batch_{folder_idx:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            out_path   = folder / f"incart_{file_idx:06d}.bin"
            win_int16.tofile(out_path)

            # ── Level 2: apakah window ini aktif? ────────────────────────────
            active      = _window_is_active(primary_class, beat_smp, beat_sym,
                                             win_start, win_end)
            class_index = primary_class if active else CLASS_TO_IDX['normal']

            if active:
                n_active += 1
            else:
                n_normal += 1

            labels.append({
                'filepath':       str(out_path),
                'class_index':    int(class_index),
                'class_name':     ARRHYTHMIA_CLASSES[class_index],
                'has_arrhythmia': int(class_index > 0),
                'source':         'incart',
                'record_name':    rec_name,
                'win_start_sec':  round(win_start / TARGET_FS, 3),
                'success':        True,
            })

            win_start += STRIDE_SAMPLES
            file_idx  += 1

        if verbose:
            pct_act = n_active / (n_active + n_normal) * 100 if (n_active+n_normal) else 0
            print(f"  [{rec_name}] {n_active+n_normal} windows: "
                  f"{n_active} active ({pct_act:.0f}%), {n_normal} normal")

        return labels, None

    except Exception as exc:
        return labels, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


# ============================================================================
# PATH UTILITIES
# ============================================================================

def _find_record_dir(rec_name: str, root: Path):
    if (root / f"{rec_name}.hea").exists():
        return root
    for sf in root.iterdir():
        if sf.is_dir():
            if (sf / f"{rec_name}.hea").exists():
                return sf
            for ssf in sf.iterdir():
                if ssf.is_dir() and (ssf / f"{rec_name}.hea").exists():
                    return ssf
    return None


# ============================================================================
# MAIN CONVERSION
# ============================================================================

def convert_incart_to_holter_format(dry_run: bool = False,
                                     verbose: bool = False) -> pd.DataFrame:
    INCART_FORMAT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ST. PETERSBURG INCART  →  HOLTER FORMAT  (v5 – ground-truth driven)")
    print("=" * 80)
    print(f"  INCART_ROOT : {INCART_ROOT}")
    print(f"  OUTPUT_DIR  : {INCART_FORMAT_DIR}")
    print(f"  Labeling    : record-descriptions.txt (ground truth) + beat annotations (window activity)")
    print(f"  Window/Stride: {WINDOW_SAMPLES}/{STRIDE_SAMPLES} samples ({WINDOW_SAMPLES/TARGET_FS:.0f}s / {STRIDE_SAMPLES/TARGET_FS:.1f}s)")
    if dry_run:
        print(f"  MODE        : DRY-RUN (rekaman pertama saja)")
    print("=" * 80)

    if not INCART_ROOT.exists():
        raise FileNotFoundError(
            f"\n✗ INCART_ROOT tidak ditemukan: {INCART_ROOT}\n"
            "  Download: https://physionet.org/content/incartdb/1.0.0/"
        )

    # ── Level 1: Parse record descriptions ───────────────────────────────────
    desc_file = INCART_ROOT / "record-descriptions.txt"
    if not desc_file.exists():
        print(f"  ⚠ record-descriptions.txt tidak ada di {INCART_ROOT}")
        print(f"    Fallback: gunakan file dari project root")
        # Coba cari di project root atau direktori ini
        for candidate in [
            PROJECT_ROOT / "record-descriptions.txt",
            Path(__file__).parent / "record-descriptions.txt",
            Path(__file__).parent.parent / "record-descriptions.txt",
        ]:
            if candidate.exists():
                desc_file = candidate
                print(f"    ✓ Found: {desc_file}")
                break

    if desc_file.exists():
        record_class_map = build_record_class_map(desc_file)
        print(f"\n  ✓ Loaded {len(record_class_map)} record descriptions")
    else:
        print(f"\n  ✗ record-descriptions.txt tidak ditemukan!")
        print(f"    Salin file ke: {INCART_ROOT}/record-descriptions.txt")
        raise FileNotFoundError(
            f"record-descriptions.txt tidak ditemukan.\n"
            f"Letakkan di: {INCART_ROOT}/"
        )

    # ── Daftar rekaman ────────────────────────────────────────────────────────
    rec_file = INCART_ROOT / "RECORDS"
    if rec_file.exists():
        record_names = [r.strip() for r in rec_file.read_text().splitlines() if r.strip()]
    else:
        record_names = [f"I{i:02d}" for i in range(1, 76)]

    if dry_run:
        record_names = record_names[:1]

    print(f"\n  Rekaman target: {len(record_names)}")

    all_labels    = []
    total_windows = 0
    errors        = {}

    for rec in tqdm(record_names, desc="Converting", unit="rec"):
        # Cari direktori file
        rec_dir = _find_record_dir(rec, INCART_ROOT)
        if rec_dir is None:
            errors[rec] = f"File {rec}.hea tidak ditemukan"
            continue

        # Primary class dari descriptions (default = normal jika tidak ada)
        primary_class = record_class_map.get(rec, CLASS_TO_IDX['normal'])

        entries, err = _process_record(
            rec_name      = rec,
            rec_dir       = rec_dir,
            primary_class = primary_class,
            out_dir       = INCART_FORMAT_DIR,
            start_idx     = total_windows,
            verbose       = verbose,
        )
        if err:
            errors[rec] = err

        all_labels.extend(entries)
        total_windows += sum(1 for e in entries if e.get('success', False))

    # ── Error report ──────────────────────────────────────────────────────────
    if errors:
        print(f"\n⚠  {len(errors)} rekaman gagal:")
        for rec, msg in list(errors.items())[:10]:
            print(f"  ✗ {rec}: {msg.splitlines()[0]}")
    else:
        print(f"\n✓ Semua rekaman berhasil diproses.")

    if not all_labels:
        print("\n✗ Tidak ada data. Jalankan --dry-run --verbose untuk debug.")
        return pd.DataFrame()

    df   = pd.DataFrame(all_labels)
    succ = df[df['success'] == True].copy()
    n    = len(succ)
    df.to_csv(INCART_LABELS_CSV, index=False)

    # ── Statistik final ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("HASIL KONVERSI INCART  (v5 ground-truth)")
    print("=" * 80)
    print(f"  Total windows    : {n:,}")

    if n == 0:
        return df

    n_arr = int(succ['has_arrhythmia'].sum())
    n_nor = n - n_arr
    print(f"  Dengan aritmia   : {n_arr:,} ({n_arr/n*100:.1f}%)")
    print(f"  Normal (window)  : {n_nor:,} ({n_nor/n*100:.1f}%)")
    print(f"\n  Distribusi per kelas:")

    stats = {
        'total_windows':   n,
        'with_arrhythmia': n_arr,
        'normal':          n_nor,
        'errors':          len(errors),
        'per_class':       {},
    }

    for idx in range(NUM_ARRHYTHMIA_CLASSES):
        name  = ARRHYTHMIA_CLASSES[idx]
        count = int((succ['class_index'] == idx).sum())
        pct   = count / n * 100
        new_m = " ← BARU dari INCART" if name in ('quadrigeminy', 'couplet', 'triplet', 'nsvt') else ""
        print(f"    [{idx:2d}] {name:25s}: {count:6,} ({pct:5.1f}%){new_m}")
        stats['per_class'][name] = {'class_index': idx, 'count': count, 'pct': round(pct, 2)}

    with open(INCART_STATS_JSON, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Labels CSV  : {INCART_LABELS_CSV}")
    print(f"✓ Stats JSON  : {INCART_STATS_JSON}")
    return df


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Konversi INCART ke Holter format v5 (ground-truth driven)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Proses hanya rekaman pertama (I01) untuk test")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detail per rekaman")
    args = parser.parse_args()

    df = convert_incart_to_holter_format(dry_run=args.dry_run, verbose=args.verbose)
    if len(df) > 0:
        ok = len(df[df['success'] == True])
        print(f"\n✅ {ok:,} windows berhasil dikonversi.")
    else:
        print("\n✗ Tidak ada output.")