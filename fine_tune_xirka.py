"""
fine_tune_xirka.py
==================
Fine-tuning model ResNet152 yang sudah dilatih (PTB-XL + INCART)
menggunakan data simulasi perangkat Xirka (ECG Holter BitActive).

Strategi:
  1. Load checkpoint terbaik (epoch 63, MacroF1=0.8499)
  2. Freeze backbone — hanya latih head classifier
  3. Dataset dari file .bin Xirka + label dari nama simulasi
  4. Augmentasi khusus karakteristik perangkat Holter wearable
  5. LR sangat kecil (1e-5) agar tidak lupakan knowledge lama

Struktur folder data Xirka yang diharapkan:
  XIRKA_DATA_DIR/
    ecg_normal/          ← atau nama apapun yang mengandung keyword
      ecg_*.bin
    ecg_premature_beat/
      ecg_*.bin
    ecg_bigeminy/
      ecg_*.bin
    ...

  ATAU letakkan semua .bin dalam satu folder dengan nama file
  yang mengandung keyword kelas (normal, bigeminy, tachycardia, dll).

Cara pakai:
  # Mode 1: folder terpisah per kelas
  python fine_tune_xirka.py --data-dir PATH/TO/XIRKA --mode folder

  # Mode 2: semua .bin dalam satu folder, nama file mengandung keyword
  python fine_tune_xirka.py --data-dir PATH/TO/XIRKA --mode filename

  # Mode 3: mapping manual via CSV
  python fine_tune_xirka.py --data-dir PATH/TO/XIRKA --mode csv \
         --csv-path xirka_labels.csv

  # Dry-run untuk verifikasi dataset sebelum training
  python fine_tune_xirka.py --data-dir PATH/TO/XIRKA --dry-run

  # Unfreeze semua layer (full fine-tuning, butuh lebih banyak data)
  python fine_tune_xirka.py --data-dir PATH/TO/XIRKA --unfreeze-all
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_path import (
    CNN_BEST_MODEL, CNN_CHECKPOINT_DIR,
    NUM_ARRHYTHMIA_CLASSES, NUM_CHANNELS, WINDOW_SIZE,
    ARRHYTHMIA_CLASSES, ARRHYTHMIA_LABELS,
    INT16_TO_MV, ADC_GAIN_DEVICE,
)
from model.resnet1d import build_model

# ── Konstanta ─────────────────────────────────────────────────────────────────
FINE_TUNE_DIR     = CNN_CHECKPOINT_DIR / "fine_tune_xirka"
FINE_TUNE_BEST    = FINE_TUNE_DIR / "ft_best_model.pth"
FINE_TUNE_LOG     = FINE_TUNE_DIR / "ft_training_log.json"

# Keyword mapping: nama file/folder → class index
# Sesuaikan dengan penamaan file simulasi Xirka Anda
KEYWORD_TO_CLASS = {
    "normal":             0,
    "premature":          1,
    "missed":             1,   # missed beat = premature beat
    "bigeminy":           2,
    "trigeminy":          3,
    "quadrigeminy":       4,
    "couplet":            5,
    "triplet":            6,
    "nsvt":               7,
    "tachycardia":        8,
    "tachy":              8,
    "bradycardia":        9,
    "brady":              9,
    "atrial_fib":        10,
    "atrial fib":        10,
    "afib":              10,
    "af":                10,
    "fibrilation":       10,   # typo di nama file Anda
    "fibrillation":      10,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_label_from_name(name: str) -> int | None:
    """
    Cari class index dari nama file atau folder.
    Return None jika tidak ada keyword yang cocok.
    """
    name_lower = name.lower()
    # Urutkan dari keyword terpanjang agar tidak ambigu
    for kw in sorted(KEYWORD_TO_CLASS, key=len, reverse=True):
        if kw in name_lower:
            return KEYWORD_TO_CLASS[kw]
    return None


def scan_xirka_files(data_dir: Path, mode: str = "folder",
                     csv_path: Path = None) -> list[dict]:
    """
    Scan folder data Xirka dan kembalikan list of dict:
      [{'path': Path, 'class_index': int, 'source': str}, ...]

    mode='folder'   : label dari nama sub-folder
    mode='filename' : label dari nama file .bin
    mode='csv'      : label dari CSV (kolom: filepath, class_index)
    """
    entries = []

    if mode == "csv":
        if csv_path is None or not csv_path.exists():
            raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            p = Path(str(row["filepath"]))
            if p.exists():
                entries.append({
                    "path":        p,
                    "class_index": int(row["class_index"]),
                    "source":      "csv",
                })
        return entries

    bin_files = sorted(data_dir.rglob("*.bin"))
    skipped   = []

    for bf in bin_files:
        if mode == "folder":
            # Cari label dari nama folder parent langsung
            label = resolve_label_from_name(bf.parent.name)
            # Fallback: coba parent-parent
            if label is None and bf.parent != data_dir:
                label = resolve_label_from_name(bf.parent.parent.name)
        else:  # filename
            label = resolve_label_from_name(bf.stem)

        if label is None:
            skipped.append(bf.name)
            continue

        entries.append({
            "path":        bf,
            "class_index": label,
            "source":      mode,
        })

    if skipped:
        print(f"  ⚠  {len(skipped)} file dilewati (tidak ada keyword): "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    return entries


class XirkaECGDataset(Dataset):
    """
    Dataset untuk fine-tuning dari file .bin perangkat Xirka.

    Format .bin Xirka  : (n_samples, 12) int16, row-major
    Faktor konversi    : × ADC_GAIN_DEVICE (0.0025) → mV
                         ATAU × INT16_TO_MV (0.001)  jika sudah di-convert

    Setiap rekaman 5 menit dipotong menjadi banyak window 5 detik.
    """

    def __init__(self,
                 entries:    list[dict],
                 window_size: int  = WINDOW_SIZE,
                 stride:      int  = 500,
                 augment:     bool = False,
                 adc_format:  str  = "device"):  # "device" | "int16"
        """
        adc_format:
          "device" → raw ADC Xirka, × 0.0025 → mV
          "int16"  → sudah ×1000 dari mV, /1000 → mV
        """
        self.window_size = window_size
        self.stride      = stride
        self.augment     = augment
        self.adc_factor  = ADC_GAIN_DEVICE if adc_format == "device" else INT16_TO_MV

        self.windows = []   # list of (path, start_sample, class_index)
        self._build_index(entries)

    def _build_index(self, entries):
        skipped = 0
        for e in entries:
            path      = e["path"]
            cls       = e["class_index"]
            file_size = path.stat().st_size
            n_samples = file_size // (NUM_CHANNELS * 2)   # int16 = 2 bytes

            if n_samples < self.window_size:
                skipped += 1
                continue

            n_win = (n_samples - self.window_size) // self.stride + 1
            for w in range(n_win):
                self.windows.append((path, w * self.stride, cls))

        if skipped:
            print(f"  ⚠  {skipped} file terlalu pendek (< {self.window_size} sampel)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        path, start, cls = self.windows[idx]

        with open(path, "rb") as f:
            f.seek(start * NUM_CHANNELS * 2)
            raw = np.fromfile(f, dtype=np.int16,
                              count=self.window_size * NUM_CHANNELS)

        if len(raw) < self.window_size * NUM_CHANNELS:
            raw = np.pad(raw, (0, self.window_size * NUM_CHANNELS - len(raw)))

        ecg = (raw.reshape(-1, NUM_CHANNELS)
                  .T
                  .astype(np.float32)) * self.adc_factor   # (12, 2500) mV

        if self.augment:
            ecg = self._augment_xirka(ecg)

        return torch.from_numpy(ecg), torch.tensor(cls, dtype=torch.long)

    # ── Augmentasi khusus karakteristik perangkat Holter wearable ─────────────

    def _augment_xirka(self, ecg: np.ndarray) -> np.ndarray:
        rng = np.random

        # 1. Amplitude scaling ±25%
        if rng.rand() > 0.5:
            ecg = ecg * rng.uniform(0.75, 1.25)

        # 2. Baseline wander (motion artifact perangkat wearable)
        #    Frekuensi rendah 0.05–0.5 Hz, amplitudo hingga 0.3 mV
        if rng.rand() > 0.4:
            t    = np.linspace(0, 5, ecg.shape[1], dtype=np.float32)
            freq = rng.uniform(0.05, 0.5)
            amp  = rng.uniform(0.02, 0.3)
            wander = amp * np.sin(2 * np.pi * freq * t)
            ecg  = ecg + wander[np.newaxis, :]

        # 3. Powerline interference 50 Hz (Indonesia)
        if rng.rand() > 0.5:
            t   = np.linspace(0, 5, ecg.shape[1], dtype=np.float32)
            amp = rng.uniform(0.005, 0.04)
            hum = amp * np.sin(2 * np.pi * 50 * t)
            ecg = ecg + hum[np.newaxis, :]

        # 4. Gaussian noise (electrode + muscle artifact)
        if rng.rand() > 0.4:
            std = rng.uniform(0.01, 0.05)
            ecg = ecg + rng.randn(*ecg.shape).astype(np.float32) * std

        # 5. Time shift circular ±200 ms = ±100 sampel
        if rng.rand() > 0.5:
            shift = rng.randint(-100, 101)
            ecg   = np.roll(ecg, shift, axis=1)

        # 6. Lead dropout (simulasi elektroda lepas)
        if rng.rand() > 0.7:
            n_drop = rng.randint(1, 3)
            leads  = rng.choice(NUM_CHANNELS, n_drop, replace=False)
            ecg[leads, :] = 0.0

        # 7. Polarity flip satu lead (elektroda terbalik)
        if rng.rand() > 0.8:
            lead = rng.randint(0, NUM_CHANNELS)
            ecg[lead, :] = -ecg[lead, :]

        # 8. DC offset kecil ±0.1 mV (kontak elektroda tidak sempurna)
        if rng.rand() > 0.5:
            ecg = ecg + rng.uniform(-0.1, 0.1)

        return ecg.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# FREEZE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def freeze_backbone(model, unfreeze_last_stage: bool = False):
    """
    Freeze semua layer kecuali head classifier.
    Opsional: unfreeze stage5 (layer ResNet terdalam) juga.

    Strategi:
      head only         → paling aman, butuh data paling sedikit
      head + stage5     → lebih fleksibel, butuh ~2x lebih banyak data
      semua layer       → full fine-tuning, butuh banyak data Xirka
    """
    # Freeze semua
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze head selalu
    for param in model.head.parameters():
        param.requires_grad = True

    # Opsional: unfreeze stage5 backbone (layer terdalam = paling high-level)
    if unfreeze_last_stage:
        # ResNet152ECG menyimpan backbone sebagai self.backbone
        if hasattr(model, "backbone") and hasattr(model.backbone, "stage5"):
            for param in model.backbone.stage5.parameters():
                param.requires_grad = True
            print("  ✓ Unfreeze: head + backbone.stage5")
        elif hasattr(model, "stage4"):
            for param in model.stage4.parameters():
                param.requires_grad = True
            print("  ✓ Unfreeze: head + stage4")
    else:
        print("  ✓ Freeze: hanya head yang dilatih")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({trainable/total*100:.1f}%)")


def unfreeze_all(model):
    """Unfreeze semua layer untuk full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Semua layer dilatih: {total:,} params")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def make_sampler(dataset: XirkaECGDataset) -> WeightedRandomSampler:
    labels  = np.array([w[2] for w in dataset.windows])
    counts  = np.bincount(labels, minlength=NUM_ARRHYTHMIA_CLASSES).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)
    weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)


def run_epoch(model, loader, criterion, device,
              optimizer=None, grad_clip: float = 1.0,
              epoch: int = 0, desc: str = "") -> dict:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    preds_all, labels_all = [], []

    bar = tqdm(loader, desc=f"  {desc}", ncols=90, leave=True)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for ecg, labels in bar:
            ecg    = ecg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(ecg)
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            preds = logits.detach().argmax(dim=1)
            total_loss  += loss.item()
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

            batch_acc = (preds.cpu() == labels.cpu()).float().mean().item()
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            acc=f"{batch_acc*100:.1f}%")

    p = np.array(preds_all)
    l = np.array(labels_all)
    all_cls = list(range(NUM_ARRHYTHMIA_CLASSES))
    return {
        "loss":         total_loss / max(len(loader), 1),
        "accuracy":     accuracy_score(l, p),
        "macro_f1":     f1_score(l, p, average="macro",
                                 zero_division=0, labels=all_cls),
        "per_class_f1": f1_score(l, p, average=None,
                                 zero_division=0, labels=all_cls).tolist(),
        "_preds":  p,
        "_labels": l,
    }


def print_metrics(phase: str, m: dict, epoch: int):
    print(f"\n  ── {phase.upper()} │ Epoch {epoch} ──")
    print(f"  Loss={m['loss']:.4f}  Acc={m['accuracy']*100:.1f}%  "
          f"MacroF1={m['macro_f1']:.4f}")
    for i, (name, f1) in enumerate(zip(ARRHYTHMIA_LABELS, m["per_class_f1"])):
        bar = "█" * int(f1 * 20)
        sym = "✓" if f1 >= 0.80 else ("!" if f1 >= 0.50 else "✗")
        print(f"  {sym} {i:2d} [{name:22s}] {f1:.3f} {bar}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FINE-TUNING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def fine_tune(args: argparse.Namespace):

    FINE_TUNE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  CPU mode")

    print("=" * 65)
    print("  FINE-TUNING ResNet152  ←  Data Simulasi Xirka")
    print("=" * 65)
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Mode label : {args.mode}")
    print(f"  ADC format : {args.adc_format}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Unfreeze   : {'semua layer' if args.unfreeze_all else 'head only' if not args.unfreeze_last_stage else 'head + last stage'}")

    # ── Scan data ─────────────────────────────────────────────────────────────
    print("\n  Scanning data Xirka...")
    entries = scan_xirka_files(
        Path(args.data_dir),
        mode     = args.mode,
        csv_path = Path(args.csv_path) if args.csv_path else None,
    )

    if not entries:
        print("\n  ✗ Tidak ada file yang ditemukan. Periksa --data-dir dan --mode.")
        print("    Contoh struktur folder yang diharapkan (mode=folder):")
        print("      data_xirka/")
        print("        ecg_normal/       → ecg_*.bin")
        print("        ecg_bigeminy/     → ecg_*.bin")
        print("        ecg_tachycardia/  → ecg_*.bin")
        return

    # Tampilkan distribusi
    print(f"\n  Total file ditemukan: {len(entries)}")
    from collections import Counter
    cls_count = Counter(e["class_index"] for e in entries)
    for idx in sorted(cls_count):
        print(f"    [{idx:2d}] {ARRHYTHMIA_CLASSES[idx]:22s}: {cls_count[idx]} file")

    if args.dry_run:
        # Hitung perkiraan window
        total_w = 0
        for e in entries:
            n = e["path"].stat().st_size // (NUM_CHANNELS * 2)
            total_w += max(0, (n - WINDOW_SIZE) // args.stride + 1)
        print(f"\n  Perkiraan total windows (stride={args.stride}): {total_w:,}")
        print("\n  [DRY-RUN] Selesai. Jalankan tanpa --dry-run untuk mulai training.")
        return

    # ── Train/Val split ───────────────────────────────────────────────────────
    cls_labels = [e["class_index"] for e in entries]
    unique_cls = list(set(cls_labels))

    # Jika kelas sangat sedikit (< 2 sampel), tidak bisa stratify
    min_cls_count = min(Counter(cls_labels).values())
    can_stratify  = min_cls_count >= 2

    if can_stratify and len(entries) >= 4:
        train_entries, val_entries = train_test_split(
            entries,
            test_size    = args.val_ratio,
            random_state = 42,
            stratify     = cls_labels,
        )
    else:
        print("  ⚠  Data terlalu sedikit untuk stratified split — split acak")
        train_entries, val_entries = train_test_split(
            entries, test_size=args.val_ratio, random_state=42
        )

    print(f"\n  Train: {len(train_entries)} file  │  Val: {len(val_entries)} file")

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    train_ds = XirkaECGDataset(
        train_entries,
        window_size = WINDOW_SIZE,
        stride      = args.stride,
        augment     = True,
        adc_format  = args.adc_format,
    )
    val_ds = XirkaECGDataset(
        val_entries,
        window_size = WINDOW_SIZE,
        stride      = WINDOW_SIZE,   # non-overlapping untuk validasi
        augment     = False,
        adc_format  = args.adc_format,
    )

    print(f"  Train windows: {len(train_ds):,}  │  Val windows: {len(val_ds):,}")

    if len(train_ds) == 0:
        print("  ✗ Tidak ada window training. Periksa file .bin dan format ADC.")
        return

    nw = min(args.num_workers, 4)
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        sampler     = make_sampler(train_ds),
        num_workers = nw,
        pin_memory  = (nw > 0),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = nw,
        pin_memory  = (nw > 0),
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(
        args.model_type,
        num_classes  = NUM_ARRHYTHMIA_CLASSES,
        num_channels = NUM_CHANNELS,
        dropout      = args.dropout,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    prev_best_f1 = ckpt.get("best_macro_f1", 0.0)
    print(f"  Checkpoint MacroF1 (PTB-XL+INCART val): {prev_best_f1:.4f}")

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────
    print()
    if args.unfreeze_all:
        unfreeze_all(model)
    else:
        freeze_backbone(model, unfreeze_last_stage=args.unfreeze_last_stage)

    model = model.to(device)

    # ── Sanity check sebelum training ─────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        sample_ecg, sample_lbl = next(iter(train_loader))
        sample_ecg = sample_ecg.to(device)
        sanity_logits = model(sample_ecg)
        sanity_loss   = nn.CrossEntropyLoss()(sanity_logits, sample_lbl.to(device))
    import math
    expected = math.log(NUM_ARRHYTHMIA_CLASSES)
    print(f"\n  ── Sanity Check ──")
    print(f"  Loss awal     : {sanity_loss.item():.4f}  (expected ≈ {expected:.4f})")
    print(f"  Label range   : {sample_lbl.min().item()} – {sample_lbl.max().item()}")
    if sanity_loss.item() > 8.0:
        print("  ⚠  Loss sangat tinggi — periksa ADC format (--adc-format)")
    elif sanity_loss.item() < 0.3:
        print("  ⚠  Loss sangat rendah — mungkin checkpoint sudah overfit ke kelas ini")
    else:
        print("  ✓  Loss normal, fine-tuning siap.")
    model.train()

    # ── Loss dengan class weights ─────────────────────────────────────────────
    # Boost kelas yang sering salah berdasarkan hasil simulasi Xirka
    class_weights = torch.ones(NUM_ARRHYTHMIA_CLASSES, dtype=torch.float32)
    class_weights[0]  = args.w_normal         # normal
    class_weights[8]  = args.w_tachy          # tachycardia (sering salah → AF)
    class_weights[10] = args.w_af             # atrial fibrillation (sering salah → Tachy)
    class_weights[4]  = args.w_quadrigeminy   # quadrigeminy (F1=0)

    print(f"\n  Class weights: normal={args.w_normal:.1f}  tachy={args.w_tachy:.1f}"
          f"  AF={args.w_af:.1f}  quad={args.w_quadrigeminy:.1f}")

    criterion = nn.CrossEntropyLoss(
        weight          = class_weights.to(device),
        label_smoothing = 0.05,
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    # LR sangat kecil untuk fine-tuning — hindari catastrophic forgetting
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = args.lr,
        weight_decay = 1e-4,
    )
    # ReduceLROnPlateau: turunkan LR jika val F1 tidak naik
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=args.lr_patience, min_lr=1e-7,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1    = 0.0
    best_epoch = 0
    no_improve = 0
    # Konversi semua Path / non-serializable ke string agar json.dump tidak error
    def _to_json_safe(v):
        if isinstance(v, Path):
            return str(v)
        return v

    log_data = {
        "epochs": [],
        "config": {k: _to_json_safe(v) for k, v in vars(args).items()},
    }

    print("\n" + "=" * 65)
    print(f"  FINE-TUNING  (epoch 1 → {args.epochs})")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'='*65}")
        print(f"  Epoch {epoch}/{args.epochs}  LR={lr:.2e}")

        train_m = run_epoch(
            model, train_loader, criterion, device,
            optimizer  = optimizer,
            grad_clip  = 1.0,
            epoch      = epoch,
            desc       = f"Train {epoch:3d}/{args.epochs}",
        )
        val_m = run_epoch(
            model, val_loader, criterion, device,
            epoch = epoch,
            desc  = f"Val   {epoch:3d}/{args.epochs}",
        )

        elapsed = time.time() - t0
        print_metrics("train", train_m, epoch)
        print_metrics("val",   val_m,   epoch)

        scheduler.step(val_m["macro_f1"])

        # Log
        log_data["epochs"].append({
            "epoch":     epoch,
            "lr":        round(lr, 9),
            "elapsed_s": round(elapsed, 1),
            "train": {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in train_m.items() if not k.startswith("_")},
            "val":   {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in val_m.items() if not k.startswith("_")},
        })
        with open(FINE_TUNE_LOG, "w") as f:
            json.dump(log_data, f, indent=2)

        # Checkpoint
        val_f1 = val_m["macro_f1"]
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch":                epoch,
                "best_macro_f1":        best_f1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "fine_tuned_from":      str(args.checkpoint),
                "xirka_data_dir":       str(args.data_dir),
            }, FINE_TUNE_BEST)
            print(f"\n  ✓ Best saved → MacroF1={best_f1:.4f}  epoch={epoch}")
        else:
            no_improve += 1
            print(f"  → No improve {no_improve}/{args.patience}")

        if no_improve >= args.patience:
            print(f"\n  ⚠  Early stop di epoch {epoch}. Best: {best_epoch}")
            break

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  FINE-TUNING SELESAI")
    print(f"  Best epoch : {best_epoch}")
    print(f"  Best MacroF1 (Xirka val) : {best_f1:.4f}")
    print(f"  Sebelumnya  (PTB-XL val) : {prev_best_f1:.4f}")
    print(f"  Checkpoint  : {FINE_TUNE_BEST}")
    print("=" * 65)

    # Classification report pada val set
    if FINE_TUNE_BEST.exists():
        print("\n  Memuat model terbaik untuk report final...")
        ckpt_ft = torch.load(FINE_TUNE_BEST, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_ft["model_state_dict"])
        final_m = run_epoch(model, val_loader, criterion, device,
                            desc="Final eval")
        p, l = final_m.pop("_preds"), final_m.pop("_labels")
        present = sorted(set(l.tolist()))
        print("\n  Classification Report (Xirka val set):")
        report = classification_report(
            l, p,
            labels      = present,
            target_names= [ARRHYTHMIA_LABELS[i] for i in present],
            zero_division= 0,
            digits      = 3,
        )
        for line in report.split("\n"):
            print(f"    {line}")

    print(f"\n  Log tersimpan : {FINE_TUNE_LOG}")
    print(f"\n  Untuk export ONNX dari model fine-tuned:")
    print(f"    python train/export_model.py export \\")
    print(f"      --checkpoint {FINE_TUNE_BEST} \\")
    print(f"      --model-type {args.model_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tuning ResNet152 ECG dengan data simulasi Xirka",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data-dir",   default=Path("C:\\Users\\Myrza\\Desktop\\project\\Project Arrythmia\\OUTPUT\\ECG Record"),
                   help="Folder berisi .bin data simulasi Xirka")
    p.add_argument("--mode",       default="folder",
                   choices=["folder", "filename", "csv"],
                   help="Cara membaca label: dari nama folder, nama file, atau CSV")
    p.add_argument("--csv-path",   default=None,
                   help="Path ke CSV jika --mode=csv (kolom: filepath, class_index)")
    p.add_argument("--adc-format", default="device",
                   choices=["device", "int16"],
                   help="device=raw ADC ×0.0025→mV | int16=×0.001→mV")
    p.add_argument("--stride",     type=int, default=500,
                   help="Stride window training (sampel)")
    p.add_argument("--val-ratio",  type=float, default=0.2,
                   help="Rasio validasi")
    p.add_argument("--num-workers",type=int, default=0,
                   help="DataLoader workers (0 untuk Windows)")

    # Model
    p.add_argument("--checkpoint",  default=str(CNN_BEST_MODEL),
                   help="Path ke checkpoint .pth yang akan di-fine-tune")
    p.add_argument("--model-type",  default="resnet152",
                   choices=["standard", "improved", "resnet152"])
    p.add_argument("--dropout",     type=float, default=0.5)
    p.add_argument("--unfreeze-all",       action="store_true",
                   help="Unfreeze semua layer (full fine-tuning)")
    p.add_argument("--unfreeze-last-stage",action="store_true",
                   help="Unfreeze head + last residual stage")

    # Training
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-5,
                   help="LR untuk fine-tuning (jauh lebih kecil dari training awal)")
    p.add_argument("--patience",    type=int,   default=10,
                   help="Early stopping patience")
    p.add_argument("--lr-patience", type=int,   default=5,
                   help="ReduceLROnPlateau patience")

    # Class weights (boost kelas yang sering salah di simulasi Xirka)
    p.add_argument("--w-normal",       type=float, default=1.0)
    p.add_argument("--w-tachy",        type=float, default=3.0)
    p.add_argument("--w-af",           type=float, default=3.0)
    p.add_argument("--w-quadrigeminy", type=float, default=5.0)

    # Util
    p.add_argument("--dry-run",    action="store_true",
                   help="Scan data dan hitung window tanpa training")

    return p.parse_args()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()
    fine_tune(args)