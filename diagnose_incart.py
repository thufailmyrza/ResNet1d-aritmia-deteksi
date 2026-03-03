"""
diagnose_incart.py
==================
Jalankan SEBELUM preprocess_incart.py untuk mendiagnosis
mengapa semua rekaman gagal diproses.

Mengecek:
  1. Apakah INCART_ROOT ada dan berisi file yang benar
  2. wfdb.rdsamp berhasil baca satu rekaman
  3. wfdb.rdann berhasil baca annotations
  4. Format info (adc_gain, sig_name, fs)
  5. Beat symbols yang ada di file
"""

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config_path import INCART_ROOT

print("=" * 70)
print("DIAGNOSA INCART DATABASE")
print("=" * 70)

# ── 1. Cek root ───────────────────────────────────────────────────────────────
print(f"\n[1] INCART_ROOT  : {INCART_ROOT}")
if not INCART_ROOT.exists():
    print("    ✗ DIREKTORI TIDAK ADA!")
    print("    → Buat folder dan letakkan file INCART di sana.")
    sys.exit(1)

all_files = list(INCART_ROOT.rglob("*.hea"))
print(f"    ✓ Ditemukan {len(all_files)} file .hea")
if not all_files:
    print("    ✗ TIDAK ADA file .hea di dalam folder!")
    sys.exit(1)

# Tampilkan 5 path pertama
for f in all_files[:5]:
    print(f"    ·  {f.relative_to(INCART_ROOT)}")

# ── 2. Temukan rekaman pertama ────────────────────────────────────────────────
first_hea = all_files[0]
rec_dir   = first_hea.parent
rec_name  = first_hea.stem           # e.g. "I01"
rec_path  = rec_dir / rec_name       # path tanpa ekstensi (untuk wfdb)

print(f"\n[2] Uji rekaman : {rec_name}")
print(f"    rec_dir     : {rec_dir}")
print(f"    rec_path    : {rec_path}")

# ── 3. wfdb.rdsamp ────────────────────────────────────────────────────────────
print("\n[3] wfdb.rdsamp ...")
try:
    import wfdb
    signal, info = wfdb.rdsamp(str(rec_path))
    print(f"    ✓ signal shape : {signal.shape}")
    print(f"    ✓ fs           : {info['fs']} Hz")
    print(f"    ✓ sig_name     : {info['sig_name']}")
    print(f"    ✓ adc_gain     : {info['adc_gain']}")
    print(f"    ✓ units        : {info.get('units', 'N/A')}")
except Exception as e:
    print(f"    ✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 4. wfdb.rdann ─────────────────────────────────────────────────────────────
print("\n[4] wfdb.rdann ...")
try:
    # Cek apakah file .atr ada
    atr_file = rec_dir / f"{rec_name}.atr"
    print(f"    .atr exists : {atr_file.exists()}  ({atr_file})")

    ann = wfdb.rdann(str(rec_path), 'atr')
    n_ann = len(ann.sample)
    symbols = sorted(set(ann.symbol))
    print(f"    ✓ Total annotations : {n_ann}")
    print(f"    ✓ Unique symbols    : {symbols}")

    # Hitung per simbol
    from collections import Counter
    cnt = Counter(ann.symbol)
    print(f"    ✓ Symbol counts     :")
    for sym, count in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"       '{sym}' : {count}")

except Exception as e:
    print(f"    ✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 5. Uji sliding window (5 window pertama) ──────────────────────────────────
print("\n[5] Uji 5 window pertama ...")
try:
    import numpy as np
    from scipy.signal import butter, filtfilt, resample as sp_resample

    INCART_FS = float(info['fs'])
    TARGET_FS = 500
    WIN       = 2500
    STRIDE    = 500

    # Scale ke mV
    mv = np.zeros_like(signal, dtype=np.float64)
    for ch in range(signal.shape[1]):
        g = float(info['adc_gain'][ch]) if info['adc_gain'][ch] else 1.0
        mv[:, ch] = signal[:, ch].astype(np.float64) / g

    print(f"    mV range  : [{mv.min():.3f}, {mv.max():.3f}]")

    # Resample
    n_tgt = int(round(mv.shape[0] * TARGET_FS / INCART_FS))
    sig500 = np.zeros((n_tgt, mv.shape[1]))
    for ch in range(mv.shape[1]):
        sig500[:, ch] = sp_resample(mv[:, ch], n_tgt)
    print(f"    After resample : {sig500.shape}  ({sig500.shape[0]/TARGET_FS:.0f}s @ {TARGET_FS}Hz)")

    # Bandpass
    nyq  = TARGET_FS / 2
    b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    filt = np.zeros_like(sig500)
    for ch in range(sig500.shape[1]):
        filt[:, ch] = filtfilt(b, a, sig500[:, ch])

    # Beat samples rescaled
    BEAT_SYM = set('NVSAaJjEeFf')
    beat_mask = np.array([s in BEAT_SYM for s in ann.symbol])
    beat_smp  = np.round(ann.sample[beat_mask] * TARGET_FS / INCART_FS).astype(int)
    beat_sym  = np.array(ann.symbol)[beat_mask]

    print(f"    Beat annotations in window: checking ...")
    n_total = filt.shape[0]
    n_win   = 0
    for ws in range(0, min(n_total - WIN + 1, 5 * STRIDE), STRIDE):
        we   = ws + WIN
        mask = (beat_smp >= ws) & (beat_smp < we)
        n_b  = mask.sum()
        syms = set(beat_sym[mask])
        print(f"    Window [{ws}..{we}] ({ws/TARGET_FS:.1f}s): {n_b} beats, symbols={syms}")
        n_win += 1

    print(f"\n    ✓ {n_win} windows diuji tanpa error")
    n_total_windows = (n_total - WIN) // STRIDE + 1
    print(f"    Estimasi total windows untuk rekaman ini: {n_total_windows}")

except Exception as e:
    print(f"    ✗ ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSA SELESAI – salin output ini untuk debugging lebih lanjut")
print("=" * 70)
