"""
train_model.py  –  v5 (fix accuracy=0 & loss meledak)
Training pipeline ECG Holter Arrhythmia Detection – 11 kelas single-label.

Perubahan dari v4:
  FIX 1: Hapus class_weights dari CrossEntropyLoss.
          WeightedRandomSampler + class_weights bersamaan menyebabkan
          sinyal gradient saling batalkan → accuracy stuck di 0.
  FIX 2: Sanity check loss awal (seharusnya ≈ ln(11) = 2.398).
  FIX 3: Training log di-reset saat mulai dari awal (bukan resume).
  FIX 4: label_smoothing default 0.1 → 0.05.

Cara pakai:
  python train_model.py                              # default
  python train_model.py --use-smote                 # aktifkan SMOTE
  python train_model.py --model-type improved        # model lebih dalam
  python train_model.py --resume                     # lanjut dari checkpoint
  python train_model.py --epochs 100 --batch-size 64 --lr 3e-4
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("C:/Users/Myrza/Desktop/project/Project Arrythmia")
sys.path.insert(0, str(PROJECT_ROOT))

from config_path import (
    HOLTER_FORMAT_DIR, NUM_CHANNELS, TRAIN_SPLIT_CSV, VAL_SPLIT_CSV, TEST_SPLIT_CSV,
    CNN_CHECKPOINT_DIR, CNN_BEST_MODEL, CNN_LAST_MODEL,
    SMOTE_CACHE_DIR, NUM_ARRHYTHMIA_CLASSES, ARRHYTHMIA_LABELS,
)
from holter_dataset import HolterECGDataset
from model.resnet1d import ResNet1D, build_model




# ============================================================================
# METRICS
# ============================================================================

class Tracker:
    """Akumulasi loss + prediksi selama satu epoch."""

    def __init__(self):
        self.total_loss = 0.0
        self.n_batches  = 0
        self.preds      = []
        self.labels     = []

    def update(self, loss: float, preds: torch.Tensor, labels: torch.Tensor):
        self.total_loss += loss
        self.n_batches  += 1
        self.preds.extend(preds.cpu().tolist())
        self.labels.extend(labels.cpu().tolist())

    def compute(self) -> dict:
        p = np.array(self.preds)
        l = np.array(self.labels)
        all_cls = list(range(NUM_ARRHYTHMIA_CLASSES))
        return {
            'loss':         self.total_loss / max(self.n_batches, 1),
            'accuracy':     accuracy_score(l, p),
            'macro_f1':     f1_score(l, p, average='macro',    zero_division=0, labels=all_cls),
            'weighted_f1':  f1_score(l, p, average='weighted', zero_division=0),
            'per_class_f1': f1_score(l, p, average=None,       zero_division=0, labels=all_cls).tolist(),
            '_preds':       p,
            '_labels':      l,
        }


def print_metrics(phase: str, m: dict, epoch: int, elapsed: float):
    print(f"\n  ── {phase.upper()} │ Epoch {epoch} │ {elapsed:.0f}s ──")
    print(f"  Loss={m['loss']:.4f}  Acc={m['accuracy']*100:.1f}%  "
          f"MacroF1={m['macro_f1']:.4f}  WtF1={m['weighted_f1']:.4f}")
    for i, (name, f1) in enumerate(zip(ARRHYTHMIA_LABELS, m['per_class_f1'])):
        bar = '█' * int(f1 * 15)
        print(f"    {i:2d} [{name:22s}] {f1:.3f} {bar}")


# ============================================================================
# TRAIN / VALIDATE
# ============================================================================

def run_epoch(model, loader, criterion, device,
              optimizer=None, grad_clip: float = 1.0,
              epoch: int = 0, total_epochs: int = 0,
              desc: str = '') -> dict:
    """
    Satu epoch training (optimizer diberikan) atau validasi (optimizer=None).
    Menampilkan tqdm progress bar dengan postfix loss & acc per batch.
    """
    is_train = optimizer is not None
    phase    = "Train" if is_train else "Val  "
    label    = desc if desc else f"Epoch {epoch:3d}/{total_epochs} [{phase}]"
    model.train() if is_train else model.eval()
    tracker = Tracker()

    bar = tqdm(
        loader,
        desc      = f"  {label}",
        ncols     = 95,
        leave     = True,
        bar_format= "{desc} {percentage:3.0f}%|{bar:25}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for ecg, labels in bar:
            ecg    = ecg.to(device, non_blocking=True)     # (B, 12, 2500) mV
            labels = labels.to(device, non_blocking=True)   # (B,) int64

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(ecg)                             # (B, 11)
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            preds = logits.detach().argmax(dim=1)
            tracker.update(loss.item(), preds, labels)

            # Postfix: running loss + batch acc
            batch_acc = (preds.cpu() == labels.cpu()).float().mean().item()
            bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc*100:.1f}%")

    return tracker.compute()


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_ckpt(model, optimizer, scheduler,
              epoch: int, best_f1: float, path: Path, tag: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'best_macro_f1':        best_f1,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, path)
    if tag:
        print(f"  ✓ Saved {tag}: {path.name}")


def load_ckpt(path: Path, model, optimizer, scheduler) -> dict:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    print(f"  ✓ Resumed epoch {ckpt['epoch']}  best F1={ckpt['best_macro_f1']:.4f}")
    return ckpt


# ============================================================================
# TRAINING LOG
# ============================================================================

def log_epoch(log_path: Path, epoch: int, train_m: dict, val_m: dict,
              lr: float, elapsed: float):
    """Append satu epoch ke training_log.json."""
    def fmt(m):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in m.items() if not k.startswith('_')}

    record = {
        'epoch': epoch, 'lr': lr, 'elapsed_s': round(elapsed, 1),
        'train': fmt(train_m), 'val': fmt(val_m),
    }
    # Baca log lama jika ada, lalu append
    if log_path.exists():
        with open(log_path) as f:
            data = json.load(f)
    else:
        data = {'epochs': []}
    data['epochs'].append(record)
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================================
# TEST SET EVALUATION
# ============================================================================

def evaluate_test(model, loader, criterion, device):
    print("\n" + "=" * 65)
    print("EVALUASI TEST SET")
    print("=" * 65)

    m = run_epoch(model, loader, criterion, device, desc='Test Set          ')
    preds  = m.pop('_preds')
    labels = m.pop('_labels')

    print(f"  Loss={m['loss']:.4f}  Acc={m['accuracy']*100:.2f}%  "
          f"MacroF1={m['macro_f1']:.4f}")

    # Classification report
    present = sorted(set(labels.tolist()))
    print("\n  Classification Report:")
    report = classification_report(
        labels, preds,
        labels=present,
        target_names=[ARRHYTHMIA_LABELS[i] for i in present],
        zero_division=0, digits=3,
    )
    for line in report.split('\n'):
        print(f"    {line}")

    # Confusion matrix (ringkas)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_ARRHYTHMIA_CLASSES)))
    print("\n  Confusion Matrix (baris=true, kolom=pred):")
    hdr = "         " + "".join(f"{i:5d}" for i in range(NUM_ARRHYTHMIA_CLASSES))
    print(f"  {hdr}")
    for i, row_data in enumerate(cm):
        row = "".join(f"{v:5d}" for v in row_data)
        print(f"  {i:2d} {ARRHYTHMIA_LABELS[i][:8]:8s} {row}")

    return m

# MAIN TRAINING FUNCTION
def train(args: argparse.Namespace):

    #  Device 
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        print("Using GPU", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
                          

    print("=" * 65)
    print("ECG Holter Arrhythmia Training  –  v4")
    print("=" * 65)
    print(f"  Device      : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"  Model       : {args.model_type}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch       : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  SMOTE       : {args.use_smote}")

    # Dataset 
    smote_dir = SMOTE_CACHE_DIR if args.use_smote else None

    train_ds = HolterECGDataset(
        labels_csv=TRAIN_SPLIT_CSV, data_root=HOLTER_FORMAT_DIR,
        stride=args.stride_train, augment=True,
        oversample_minority=True, smote_npy_dir=smote_dir,
    )
    val_ds = HolterECGDataset(
        labels_csv=VAL_SPLIT_CSV, data_root=HOLTER_FORMAT_DIR,
        stride=2500, augment=False,
        oversample_minority=False, smote_npy_dir=None,   # ← tidak pakai SMOTE
    )
    test_ds = HolterECGDataset(
        labels_csv=TEST_SPLIT_CSV, data_root=HOLTER_FORMAT_DIR,
        stride=2500, augment=False,
        oversample_minority=False, smote_npy_dir=None,   # ← tidak pakai SMOTE
    )

    print(f"\n  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    for name, cnt in train_ds.class_distribution().items():
        print(f"    {name:22s}: {cnt:,}")

    # Windows
    nw = args.num_workers

    def make_loader(ds, shuffle=False, sampler=None, drop_last=False):
        return DataLoader(ds, batch_size=args.batch_size,
                          sampler=sampler, shuffle=shuffle,
                          num_workers=nw, pin_memory=(nw > 0),
                          persistent_workers=(nw > 0), drop_last=drop_last)

    train_loader = make_loader(train_ds, sampler=train_ds.get_sampler(), drop_last=True)
    val_loader   = make_loader(val_ds)
    test_loader  = make_loader(test_ds)

    # Model 
    model = build_model(
        args.model_type,
        dropout=args.dropout,
        num_channels=NUM_CHANNELS,
        num_classes=NUM_ARRHYTHMIA_CLASSES
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Params: {n_params:,}")

    # ── Loss 
    criterion = nn.CrossEntropyLoss(
        label_smoothing = args.label_smoothing,
    )
    # Sanity Check: Verifikasi loss awal 
    # Dengan random init & 11 kelas, loss seharusnya ≈ ln(11) = 2.398
    # Jika jauh berbeda, ada masalah data/label/model.
    model.eval()
    with torch.no_grad():
        sample_ecg, sample_lbl = next(iter(train_loader))
        sample_ecg = sample_ecg.to(device)
        sample_lbl = sample_lbl.to(device)
        sanity_logits = model(sample_ecg)
        sanity_loss   = nn.CrossEntropyLoss()(sanity_logits, sample_lbl)
        expected_loss = __import__('math').log(NUM_ARRHYTHMIA_CLASSES)
        print(f"\n  ── Sanity Check ──")
        print(f"  Loss awal     : {sanity_loss.item():.4f}")
        print(f"  Expected (≈)  : {expected_loss:.4f}  [ln({NUM_ARRHYTHMIA_CLASSES})]")
        print(f"  Label range   : {sample_lbl.min().item()}–{sample_lbl.max().item()}")
        print(f"  Logit range   : {sanity_logits.min().item():.3f}–{sanity_logits.max().item():.3f}")
        if sanity_loss.item() < 0.5 or sanity_loss.item() > 10.0:
            print(f"  ⚠ WARNING: Loss tidak normal! Periksa label/data/model init.")
        else:
            print(f"  ✓ Loss normal, training siap.")
    model.train()

    # Optimizer 
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler 
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.t_max, T_mult=2, eta_min=1e-6)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.patience, min_lr=1e-6)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    # Resume 
    start_epoch   = 1
    best_f1       = 0.0
    best_epoch    = 0
    no_improve    = 0

    if args.resume and CNN_BEST_MODEL.exists():
        ckpt       = load_ckpt(CNN_BEST_MODEL, model, optimizer, scheduler)
        start_epoch = ckpt['epoch'] + 1
        best_f1    = ckpt['best_macro_f1']
        best_epoch = ckpt['epoch']
    elif args.resume:
        print("  ⚠ Checkpoint tidak ditemukan, mulai dari awal.")

    log_path = CNN_CHECKPOINT_DIR / "training_log.json"
    CNN_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Buat log baru saat mulai dari awal (bukan resume) – hindari log lama tercampur
    if not args.resume or not log_path.exists():
        config_dict = vars(args).copy()
        config_dict['model_type']    = args.model_type
        config_dict['num_classes']   = NUM_ARRHYTHMIA_CLASSES
        config_dict['num_channels']  = NUM_CHANNELS
        config_dict['window_size']   = 2500
        config_dict['stride_train']  = args.stride_train
        config_dict['stride_val']    = 2500
        with open(log_path, 'w') as f:
            json.dump({
                'config':     config_dict,
                'start_time': datetime.now().isoformat(),
                'epochs':     [],
            }, f, indent=2)
        print(f"  ✓ Log baru dibuat: {log_path}")

    # Training Loop 
    print("\n" + "=" * 65)
    print(f"TRAINING  (epoch {start_epoch} → {args.epochs})")
    print("=" * 65)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*65}")
        print(f"  Epoch {epoch}/{args.epochs}  LR={lr:.2e}")

        train_m = run_epoch(model, train_loader, criterion, device,
                            optimizer=optimizer,
                            grad_clip=args.grad_clip,
                            epoch=epoch, total_epochs=args.epochs)

        val_m = run_epoch(model, val_loader, criterion, device,
                          epoch=epoch, total_epochs=args.epochs)

        elapsed = time.time() - t0
        print_metrics('train', train_m, epoch, elapsed)
        print_metrics('val',   val_m,   epoch, elapsed)

        # Scheduler step
        if args.scheduler == 'plateau':
            scheduler.step(val_m['macro_f1'])
        else:
            scheduler.step()

        # Log
        log_epoch(log_path, epoch, train_m, val_m, lr, elapsed)

        # Best checkpoint
        val_f1 = val_m['macro_f1']
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            no_improve = 0
            save_ckpt(model, optimizer, scheduler,
                      epoch, best_f1, CNN_BEST_MODEL,
                      tag=f"best (MacroF1={val_f1:.4f})")
        else:
            no_improve += 1
            print(f"  → No improve {no_improve}/{args.early_stop_patience}")

        # Last checkpoint tiap N epoch
        if epoch % args.save_every == 0:
            save_ckpt(model, optimizer, scheduler,
                      epoch, best_f1, CNN_LAST_MODEL, tag=f"last (epoch {epoch})")

        # Early stopping
        if no_improve >= args.early_stop_patience:
            print(f"\n  ⚠ Early stop di epoch {epoch}. Best: {best_epoch} F1={best_f1:.4f}")
            break

        print(f"  ✓ Best epoch={best_epoch}  MacroF1={best_f1:.4f}")

    # Final Summary
    print("\n" + "=" * 65)
    print(f"SELESAI  –  Best epoch={best_epoch}  MacroF1={best_f1:.4f}")
    print("=" * 65)

    # Update log dengan info akhir
    if log_path.exists():
        with open(log_path) as f: data = json.load(f)
    else:
        data = {'epochs': []}
    data.update({'best_epoch': best_epoch, 'best_macro_f1': round(best_f1, 4),
                 'end_time': datetime.now().isoformat()})
    with open(log_path, 'w') as f: json.dump(data, f, indent=2)

    # Test Evaluation 
    if CNN_BEST_MODEL.exists():
        ckpt = torch.load(CNN_BEST_MODEL, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    test_m = evaluate_test(model, test_loader, criterion, device)

    result_path = CNN_CHECKPOINT_DIR / "test_results.json"
    with open(result_path, 'w') as f:
        json.dump({'best_epoch': best_epoch,
                   'metrics': {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in test_m.items()},
                   'timestamp': datetime.now().isoformat()}, f, indent=2)
    print(f"\n  ✓ Test results: {result_path}")

    return model


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='ECG Holter Arrhythmia Detection Training v4',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument('--model-type',  default='standard', choices=['standard', 'improved'])
    p.add_argument('--dropout',     type=float, default=0.3)
    # Data
    p.add_argument('--stride-train',type=int,   default=500,
                   help='Stride window training (sampel; 500=80%% overlap)')
    p.add_argument('--use-smote', default=True, action='store_true', help='Aktifkan SMOTE synthetic windows')
    p.add_argument('--num-workers', type=int,   default=4,
                   help='DataLoader workers (paksa 0 di Windows)')
    # Training
    p.add_argument('--epochs',      type=int,   default=80)
    p.add_argument('--batch-size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--weight-decay',type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.05,
                   help='Label smoothing (0=off, 0.05 recommended, jangan >0.1)')
    p.add_argument('--grad-clip',   type=float, default=1.0)
    # Scheduler
    p.add_argument('--scheduler',   default='cosine', choices=['cosine', 'plateau', 'step'])
    p.add_argument('--t-max',       type=int,   default=30, help='CosineAnnealing period')
    p.add_argument('--patience',    type=int,   default=10, help='ReduceLROnPlateau patience')
    # Checkpoint
    p.add_argument('--resume',      action='store_true', help='Resume dari best_model.pth')
    p.add_argument('--early-stop-patience', type=int, default=20)
    p.add_argument('--save-every',  type=int,   default=5, help='Simpan last_model tiap N epoch')

    args = p.parse_args()
    return args


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == '__main__':
    args = parse_args()

    # Seed
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
        print("✓ Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU")
    
    train(args)