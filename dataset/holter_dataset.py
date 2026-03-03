"""
holter_dataset.py
Dataset PyTorch untuk ECG Holter – single-label 11 kelas.

Perubahan dari v3:
  1. Bug fix: smote_windows/smote_labels diinisialisasi None (bukan boolean True).
  2. Bug fix: get_class_weights() dan get_sampler() dihitung dari window REAL saja
     (bukan termasuk SMOTE synthetic). SMOTE menggelembungkan kelas minoritas
     sehingga kelas Normal tampak "langka" → weight Normal naik drastis → loss meledak.
  3. Normalisasi weights: mean=1.0 (lebih stabil vs sum=N_classes).
  4. n_real_windows disimpan untuk memisahkan real vs synthetic.
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_path import (
    NUM_ARRHYTHMIA_CLASSES,   # 11
    NUM_CHANNELS,             # 12
    ARRHYTHMIA_CLASSES,       # dict idx→name
    ARRHYTHMIA_LABELS,        # list[str], indeks = class idx
    INT16_TO_MV,              # = 1/1000 = 0.001
    WINDOW_SIZE,              # 2500
    SMOTE_CACHE_DIR,
)

# DATASET UTAMA  –  ORIGINAL + OPTIONAL SMOTE SYNTHETIC
class HolterECGDataset(Dataset):
    """
    Dataset ECG Holter, single-label (11 kelas), skala mV.

    Setiap item yang dikembalikan:
      ecg   : torch.Tensor shape (12, WINDOW_SIZE) float32  –  mV scale
      label : torch.LongTensor scalar  –  class index 0–10

    Args:
        labels_csv:           Path ke CSV split (train/val/test).
        data_root:            Root folder data biner Holter.
        window_size:          Panjang window dalam sampel (default WINDOW_SIZE=2500).
        stride:               Stride antar window (sampel).
        augment:              Aktifkan augmentasi.
        oversample_minority:  Buat lebih banyak window dari rekaman aritmia.
        smote_npy_dir:        Folder berisi synthetic_windows.npy &
                              synthetic_labels.npy dari SMOTE (None = tidak pakai).
    """

    def __init__(self, labels_csv, data_root,
                 window_size: int = WINDOW_SIZE,
                 stride:      int = 500,
                 augment:     bool = False,
                 oversample_minority: bool = True,
                 smote_npy_dir=SMOTE_CACHE_DIR):

        self.labels_df          = pd.read_csv(labels_csv)
        self.data_root          = Path(data_root)
        self.window_size        = window_size
        self.stride             = stride
        self.augment            = augment
        self.oversample_minority = oversample_minority

        # Bangun indeks window dari rekaman asli
        self._build_window_index()

        # Catat jumlah window real (sebelum SMOTE) untuk class weights
        self.n_real_windows = len(self.window_index)

        # Tambahkan synthetic SMOTE windows jika ada
        self.smote_windows = None   # None = belum diload
        self.smote_labels  = None
        if smote_npy_dir is not None:
            self._load_smote_windows(Path(smote_npy_dir))

        # Hitung sampling weights untuk WeightedRandomSampler
        self._compute_sample_weights()

    # Bangun Indeks Window 
    @staticmethod
    def _resolve_bin_path(row, data_root: Path):
        """
        Resolve path ke file .bin dari baris CSV.

        Mendukung dua format kolom:
          Format A (PTB-XL)  : batch_dir + output_filename  → data_root/batch_dir/output_filename
          Format B (INCART)  : filepath (path absolut)      → langsung dipakai

        Urutan coba:
          1. filepath absolut (INCART)
          2. data_root / batch_dir / output_filename (PTB-XL)
          3. INCART_FORMAT_DIR / batch_dir / output_filename (jika source=incart)

        Returns Path atau None jika tidak bisa di-resolve.
        """
        from config_path import INCART_FORMAT_DIR, HOLTER_FORMAT_DIR

        # Prioritas 1: filepath absolut (INCART rows)
        fp = row.get('filepath', None)
        if fp is not None and str(fp) not in ('', 'nan'):
            p = Path(str(fp))
            if p.exists():
                return p

        # Prioritas 2: data_root / batch_dir / output_filename (PTB-XL rows)
        batch_dir = row.get('batch_dir', None)
        out_fname = row.get('output_filename', None)
        if (batch_dir is not None and str(batch_dir) not in ('', 'nan')
                and out_fname is not None and str(out_fname) not in ('', 'nan')):
            # Tentukan root yang benar berdasarkan source atau nama file
            source = str(row.get('source', '')).lower()
            fname  = str(out_fname)

            if source == 'incart' or fname.startswith('incart_'):
                # INCART binary ada di INCART_FORMAT_DIR
                p = INCART_FORMAT_DIR / str(batch_dir) / fname
            else:
                # PTB-XL binary ada di data_root (HOLTER_FORMAT_DIR)
                p = data_root / str(batch_dir) / fname

            if p.exists():
                return p

            # Last resort: coba keduanya
            for root in [data_root, INCART_FORMAT_DIR, HOLTER_FORMAT_DIR]:
                p = root / str(batch_dir) / fname
                if p.exists():
                    return p

        return None  # tidak bisa di-resolve

    @staticmethod
    def _resolve_class_label(row) -> int:
        """
        Resolve class label dari baris CSV.

        Mendukung dua nama kolom:
          'class_label'  → PTB-XL format
          'class_index'  → INCART / merged format
        """
        if 'class_label' in row and str(row['class_label']) not in ('', 'nan'):
            return int(row['class_label'])
        if 'class_index' in row and str(row['class_index']) not in ('', 'nan'):
            return int(row['class_index'])
        return 0  # fallback normal

    def _build_window_index(self):
        """
        Bangun list window dari semua rekaman di labels_csv.
        Rekaman dengan kelas minoritas mendapat stride lebih kecil
        (= lebih banyak window) jika oversample_minority=True.

        Mendukung kolom CSV dari PTB-XL (batch_dir + output_filename + class_label)
        maupun INCART / merged (filepath + class_index).
        """
        self.window_index = []  # list of dict

        # Tentukan kolom class yang tersedia
        has_class_label = 'class_label' in self.labels_df.columns
        has_class_index = 'class_index' in self.labels_df.columns
        if not has_class_label and not has_class_index:
            raise KeyError(
                "CSV tidak memiliki kolom 'class_label' maupun 'class_index'.\n"
                "Jalankan merge_dataset.py untuk memastikan kolom terstandarisasi."
            )

        # Hitung jumlah rekaman per kelas untuk oversample factor
        cls_col      = 'class_label' if has_class_label else 'class_index'
        class_counts = self.labels_df[cls_col].value_counts()
        max_count    = class_counts.max()

        skipped = 0
        for _, row in self.labels_df.iterrows():
            bin_path = self._resolve_bin_path(row, self.data_root)
            if bin_path is None or not bin_path.exists():
                skipped += 1
                continue

            cls        = self._resolve_class_label(row)
            n_samples  = bin_path.stat().st_size // (NUM_CHANNELS * 2)

            if self.oversample_minority and cls != 0:
                # Semakin minoritas → stride lebih kecil → lebih banyak window
                cls_count = class_counts.get(cls, 1)
                ratio     = min(max_count / max(cls_count, 1), 8.0)
                stride    = max(125, int(self.stride / ratio))
            else:
                stride    = self.stride

            n_win = max(1, (n_samples - self.window_size) // stride + 1)

            for w in range(n_win):
                self.window_index.append({
                    'bin_path':    bin_path,
                    'start':       w * stride,
                    'class_label': cls,
                    'is_real':     True,    # bukan synthetic
                })

        if skipped > 0:
            print(f"  ⚠ {skipped} baris CSV dilewati (file tidak ditemukan atau path invalid)")

    # Load SMOTE Synthetic Windows 

    def _load_smote_windows(self, smote_dir: Path):
        """
        Load synthetic windows dari output smote_oversampling.py.

        File yang diharapkan:
          synthetic_windows.npy  → shape (N, 12, window_size) float32 (mV)
          synthetic_labels.npy   → shape (N,) int64
        """
        w_path = smote_dir / "synthetic_windows.npy"
        l_path = smote_dir / "synthetic_labels.npy"

        if not w_path.exists() or not l_path.exists():
            print(f"  ⚠ SMOTE files tidak ditemukan di {smote_dir}, dilewati.")
            return

        self.smote_windows = np.load(w_path, mmap_mode='r')  # (N, 12, W)
        self.smote_labels  = np.load(l_path)                  # (N,)

        # Tambahkan ke window_index
        for i in range(len(self.smote_labels)):
            self.window_index.append({
                'bin_path':    None,
                'start':       i,            # dipakai sebagai indeks ke smote array
                'class_label': int(self.smote_labels[i]),
                'is_real':     False,        # synthetic
            })

        print(f"  ✓ Loaded {len(self.smote_labels):,} synthetic SMOTE windows")

    # Sampling Weights 

    def _compute_sample_weights(self):
        """
        Hitung per-sample weight untuk WeightedRandomSampler.
        Weight = 1 / freq_class  (inverse frequency).
        """
        labels = np.array([w['class_label'] for w in self.window_index])
        counts = np.bincount(labels, minlength=NUM_ARRHYTHMIA_CLASSES).astype(float)
        counts = np.where(counts == 0, 1.0, counts)   # hindari div-by-zero
        inv_freq = 1.0 / counts
        self.sample_weights = inv_freq[labels].tolist()

    # Dataset Interface 

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int):
        info = self.window_index[idx]
        cls  = info['class_label']

        if info['is_real']:
            ecg = self._read_real_window(info['bin_path'], info['start'])
        else:
            # Synthetic SMOTE window (sudah dalam mV, shape 12×W)
            ecg = self.smote_windows[info['start']].copy().astype(np.float32)

        if self.augment:
            ecg = self._augment(ecg)

        return torch.from_numpy(ecg), torch.tensor(cls, dtype=torch.long)

    # Baca Window dari .bin 
    def _read_real_window(self, bin_path: Path,
                          start_sample: int) -> np.ndarray:
        """
        Baca window ECG dari file .bin dan normalisasi ke mV.

        Returns:
            np.ndarray shape (12, window_size) float32  –  satuan mV
        """
        with open(bin_path, 'rb') as f:
            f.seek(start_sample * NUM_CHANNELS * 2)
            raw = np.fromfile(f, dtype=np.int16,
                              count=self.window_size * NUM_CHANNELS)

        if len(raw) < self.window_size * NUM_CHANNELS:
            raw = np.pad(raw,
                         (0, self.window_size * NUM_CHANNELS - len(raw)),
                         mode='constant')

        ecg = raw.reshape(-1, NUM_CHANNELS).T   # (12, window_size)
        # int16 / 1000 = mV  (konsisten dengan ecg_signal_mv di app)
        return ecg.astype(np.float32) * INT16_TO_MV

    # Augmentasi 
    def _augment(self, ecg: np.ndarray) -> np.ndarray:
        """
        Augmentasi pada data mV-scale.
        Semua operasi mempertahankan skala mV agar konsisten dengan app.
        """
        rng = np.random

        # 1. Amplitude scaling (±30%)
        if rng.rand() > 0.5:
            ecg = ecg * rng.uniform(0.7, 1.3)

        # 2. Baseline wander (shift kecil dalam mV, ±0.1 mV)
        if rng.rand() > 0.5:
            ecg = ecg + rng.uniform(-0.1, 0.1)

        # 3. Gaussian noise (std 0.02 mV, realistis untuk noise elektroda)
        if rng.rand() > 0.5:
            ecg = ecg + rng.randn(*ecg.shape).astype(np.float32) * 0.02

        # 4. Time shift (circular, ±200ms = ±100 sampel)
        if rng.rand() > 0.5:
            shift = rng.randint(-100, 100)
            ecg   = np.roll(ecg, shift, axis=1)

        # 5. Lead dropout (1–2 lead di-nol-kan)
        if rng.rand() > 0.7:
            n    = rng.randint(1, 3)
            drop = rng.choice(NUM_CHANNELS, n, replace=False)
            ecg[drop, :] = 0.0

        # 6. Polarity flip (simulasi elektroda terbalik, khusus lead tertentu)
        if rng.rand() > 0.8:
            lead = rng.randint(0, NUM_CHANNELS)
            ecg[lead, :] = -ecg[lead, :]

        return ecg.astype(np.float32)

    #  Utility 
    def get_sampler(self) -> WeightedRandomSampler:
        """Return WeightedRandomSampler berbasis inverse class frequency (real data only)."""
        # Gunakan label real untuk hitung sampling weight
        real_labels = np.array([
            w['class_label'] for w in self.window_index[:self.n_real_windows]
        ])
        counts   = np.bincount(real_labels, minlength=NUM_ARRHYTHMIA_CLASSES).astype(float)
        counts   = np.where(counts == 0, 1.0, counts)
        inv_freq = 1.0 / counts

        # Weight per sample = inverse freq kelas-nya
        all_labels = np.array([w['class_label'] for w in self.window_index])
        sample_w   = inv_freq[all_labels]
        return WeightedRandomSampler(
            weights    = sample_w.tolist(),
            num_samples= len(sample_w),
            replacement= True
        )

    def get_class_weights(self) -> torch.Tensor:
        """
        Return class weights tensor untuk CrossEntropyLoss(weight=...).
        Shape: (NUM_ARRHYTHMIA_CLASSES,) float32.

        PENTING: Dihitung dari window REAL saja (bukan SMOTE synthetic).
        Ini mencegah distorsi ekstrem: SMOTE menambah ratusan ribu synthetic
        windows ke kelas minoritas, sehingga kelas Normal menjadi "langka"
        di window_index dan mendapat weight sangat besar → loss tidak stabil.
        """
        # Hanya window real (sebelum SMOTE ditambahkan)
        real_labels = np.array([
            w['class_label'] for w in self.window_index[:self.n_real_windows]
        ])
        counts  = np.bincount(real_labels, minlength=NUM_ARRHYTHMIA_CLASSES).astype(float)
        counts  = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        # Normalisasi: mean weight = 1.0 agar loss tetap dalam skala normal
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    def class_distribution(self) -> dict:
        """Return distribusi window per kelas."""
        labels = [w['class_label'] for w in self.window_index]
        counts = {}
        for cls in range(NUM_ARRHYTHMIA_CLASSES):
            cnt = labels.count(cls)
            if cnt > 0:
                counts[ARRHYTHMIA_CLASSES[cls]] = cnt
        return counts

    def __repr__(self) -> str:
        dist = self.class_distribution()
        lines = [f"HolterECGDataset(n={len(self)}, augment={self.augment})"]
        for name, cnt in dist.items():
            lines.append(f"  {name:22s}: {cnt:,}")
        return '\n'.join(lines)

# DATASET UNTUK INFERENCE  –  streaming tanpa label
class HolterInferenceDataset(Dataset):
    """
    Dataset untuk inference pada rekaman Holter panjang (streaming window).

    Input: satu file .bin (raw int16) atau array mV.
    Output: (ecg_tensor, start_sample_idx) per window.

    Dipakai oleh inference_export_pkl.py untuk feed ke model.
    """

    def __init__(self, ecg_mv: np.ndarray, window_size: int = WINDOW_SIZE,
                 stride: int = WINDOW_SIZE):
        """
        Args:
            ecg_mv:      ECG dalam mV, shape (n_samples, 12) atau (12, n_samples).
            window_size: Panjang window (sampel).
            stride:      Stride antar window (sampel).
                         Default = window_size (tidak overlap, cocok dengan app).
        """
        if ecg_mv.ndim == 2 and ecg_mv.shape[1] == NUM_CHANNELS:
            ecg_mv = ecg_mv.T   # (12, n_samples)

        assert ecg_mv.shape[0] == NUM_CHANNELS, \
            f"Expected {NUM_CHANNELS} channels, got {ecg_mv.shape[0]}"

        self.ecg_mv       = ecg_mv.astype(np.float32)
        self.window_size  = window_size
        self.stride       = stride
        self.n_samples    = ecg_mv.shape[1]

        starts = list(range(0, self.n_samples - window_size + 1, stride))
        # Pastikan sampel terakhir selalu ada
        if not starts or starts[-1] + window_size < self.n_samples:
            starts.append(max(0, self.n_samples - window_size))
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s   = self.starts[idx]
        win = self.ecg_mv[:, s:s + self.window_size]
        if win.shape[1] < self.window_size:
            pad = np.zeros((NUM_CHANNELS, self.window_size - win.shape[1]),
                           dtype=np.float32)
            win = np.concatenate([win, pad], axis=1)
        return torch.from_numpy(win), s