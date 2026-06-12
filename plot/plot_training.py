"""
plot_training.py
================
Plot hasil training dari training_log.json.

Layout (6 panel):
  1. Train vs Val Loss
  2. Train vs Val Macro F1
  3. Train vs Val Accuracy
  4. Train vs Val Weighted F1
  5. Learning Rate schedule
  6. Per-class F1 (best epoch) — bar chart

Cara pakai:
  python plot_training.py --log OUTPUT/checkpoints/cnn/training_log.json
  python plot_training.py --log training_log.json --save hasil_training.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config_path import (
    CNN_TRAINING_LOG
)
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':    9.5,
    'axes.titlesize': 11,
    'axes.labelsize': 9.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

ARRHYTHMIA_LABELS = [
    'Normal', 'Premature Beat', 'Bigeminy', 'Trigeminy',
    'Quadrigeminy', 'Couplet', 'Triplet', 'NSVT',
    'Tachycardia', 'Bradycardia', 'Atrial Fib.',
]

CLASS_COLORS = [
    '#95A5A6','#3498DB','#2ECC71','#1ABC9C','#27AE60',
    '#F39C12','#E67E22','#E74C3C','#C0392B','#8E44AD','#2C3E50',
]

C_TRAIN  = '#2980B9'
C_VAL    = '#E74C3C'
C_BEST   = '#27AE60'
C_LR     = '#F39C12'
C_BG     = '#F8F9FA'
C_GRID   = '#E0E0E0'


# ============================================================
def load_log(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {p}")
    with open(p) as f:
        data = json.load(f)
    if not data.get('epochs'):
        raise ValueError("training_log.json tidak mengandung data epoch.")
    return data


def extract(epochs_data: list) -> dict:
    out = {
        'epoch':        [],
        'train_loss':   [], 'val_loss':   [],
        'train_acc':    [], 'val_acc':    [],
        'train_mf1':    [], 'val_mf1':    [],
        'train_wf1':    [], 'val_wf1':    [],
        'lr':           [],
        'train_pcf1':   [],   # per-class F1 setiap epoch
        'val_pcf1':     [],
        'elapsed':      [],
    }
    for d in epochs_data:
        out['epoch'].append(d['epoch'])
        out['lr'].append(d.get('lr', 0))
        out['elapsed'].append(d.get('elapsed_s', 0))

        t = d.get('train', {})
        v = d.get('val',   {})

        out['train_loss'].append(t.get('loss', np.nan))
        out['val_loss'].append(  v.get('loss', np.nan))

        out['train_acc'].append(t.get('accuracy', np.nan))
        out['val_acc'].append(  v.get('accuracy', np.nan))

        out['train_mf1'].append(t.get('macro_f1',    np.nan))
        out['val_mf1'].append(  v.get('macro_f1',    np.nan))

        out['train_wf1'].append(t.get('weighted_f1', np.nan))
        out['val_wf1'].append(  v.get('weighted_f1', np.nan))

        out['train_pcf1'].append(t.get('per_class_f1', [np.nan]*11))
        out['val_pcf1'].append(  v.get('per_class_f1', [np.nan]*11))

    # numpy arrays
    for k in ['epoch','train_loss','val_loss','train_acc','val_acc',
              'train_mf1','val_mf1','train_wf1','val_wf1','lr','elapsed']:
        out[k] = np.array(out[k], dtype=float)
    out['train_pcf1'] = np.array(out['train_pcf1'], dtype=float)   # (E, 11)
    out['val_pcf1']   = np.array(out['val_pcf1'],   dtype=float)

    return out


# ============================================================
def _style_ax(ax, title, xlabel='Epoch', ylabel=''):
    ax.set_facecolor(C_BG)
    ax.set_title(title, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _vline_best(ax, best_ep, ymin, ymax, label=True):
    ax.axvline(best_ep, color=C_BEST, ls=':', lw=1.8, zorder=4,
               label=f'Best ep={int(best_ep)}' if label else None)


def plot_loss(ax, d, best_ep):
    ax.plot(d['epoch'], d['train_loss'], color=C_TRAIN, lw=2,   label='Train')
    ax.plot(d['epoch'], d['val_loss'],   color=C_VAL,   lw=2, ls='--', label='Val')
    _vline_best(ax, best_ep, 0, 1)
    _style_ax(ax, 'Cross-Entropy Loss', ylabel='Loss')
    ax.legend(fontsize=8)


def plot_macro_f1(ax, d, best_ep):
    ax.plot(d['epoch'], d['train_mf1'], color=C_TRAIN, lw=2,   label='Train')
    ax.plot(d['epoch'], d['val_mf1'],   color=C_VAL,   lw=2, ls='--', label='Val')
    # Mark best value
    best_idx = int(np.nanargmax(d['val_mf1']))
    best_val = d['val_mf1'][best_idx]
    ax.scatter(d['epoch'][best_idx], best_val, s=80, color=C_BEST, zorder=6)
    ax.annotate(f'  {best_val:.4f}', xy=(d['epoch'][best_idx], best_val),
                fontsize=8.5, color=C_BEST, fontweight='bold', va='bottom')
    _vline_best(ax, best_ep, 0, 1)
    _style_ax(ax, 'Macro F1-Score', ylabel='Macro F1')
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1]))
    ax.legend(fontsize=8)


def plot_accuracy(ax, d, best_ep):
    ax.plot(d['epoch'], d['train_acc'] * 100, color=C_TRAIN, lw=2,   label='Train')
    ax.plot(d['epoch'], d['val_acc']   * 100, color=C_VAL,   lw=2, ls='--', label='Val')
    _vline_best(ax, best_ep, 0, 100)
    _style_ax(ax, 'Accuracy (%)', ylabel='Accuracy (%)')
    ax.legend(fontsize=8)


def plot_weighted_f1(ax, d, best_ep):
    ax.plot(d['epoch'], d['train_wf1'], color=C_TRAIN, lw=2,   label='Train')
    ax.plot(d['epoch'], d['val_wf1'],   color=C_VAL,   lw=2, ls='--', label='Val')
    _vline_best(ax, best_ep, 0, 1)
    _style_ax(ax, 'Weighted F1-Score', ylabel='Weighted F1')
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1]))
    ax.legend(fontsize=8)


def plot_lr(ax, d):
    ax.semilogy(d['epoch'], d['lr'], color=C_LR, lw=2)
    _style_ax(ax, 'Learning Rate Schedule', ylabel='LR (log scale)')
    ax.fill_between(d['epoch'], d['lr'], alpha=0.15, color=C_LR)


def plot_per_class_f1(ax, d, best_ep):
    # Cari indeks epoch best
    best_idx = int(np.nanargmax(d['val_mf1']))
    val_pcf1 = d['val_pcf1'][best_idx]   # (11,)

    n = len(ARRHYTHMIA_LABELS)
    x = np.arange(n)
    bars = ax.barh(x, val_pcf1, color=CLASS_COLORS, edgecolor='white',
                   linewidth=0.8, height=0.7, zorder=3)

    # Nilai di ujung bar
    for i, (bar, v) in enumerate(zip(bars, val_pcf1)):
        if not np.isnan(v):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8,
                    color='#333', fontweight='bold')

    ax.set_yticks(x)
    ax.set_yticklabels(ARRHYTHMIA_LABELS, fontsize=8.5)
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color='#BDBDBD', ls='--', lw=1, zorder=2)
    ax.set_facecolor(C_BG)
    ax.grid(True, axis='x', color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f'Per-class F1  (Val — Best epoch {int(best_ep)})',
                 fontweight='bold', pad=6)
    ax.set_xlabel('F1-Score')


# ============================================================
#  GENERATE DUMMY DATA  (untuk demo tanpa log nyata)
# ============================================================
def make_dummy_log(n_epochs=60) -> dict:
    np.random.seed(42)
    eps = np.arange(1, n_epochs + 1)
    lr_base = 1e-3
    data = {'config': {'model_type': 'standard (DEMO)', 'epochs': n_epochs},
            'start_time': '2025-01-01T00:00:00',
            'epochs': [], 'best_epoch': 0, 'best_macro_f1': 0.0}

    best_f1, best_ep = 0.0, 1
    for ep in eps:
        t = ep / n_epochs
        tr_loss = 2.4 * np.exp(-3.5 * t) + 0.25 + np.random.normal(0, 0.04)
        va_loss = 2.4 * np.exp(-3.0 * t) + 0.35 + np.random.normal(0, 0.05)
        tr_acc  = min(0.98, 0.20 + 0.75 * (1 - np.exp(-5*t)) + np.random.normal(0, 0.01))
        va_acc  = min(0.95, 0.18 + 0.70 * (1 - np.exp(-4*t)) + np.random.normal(0, 0.015))
        tr_mf1  = min(0.97, 0.05 + 0.88 * (1 - np.exp(-5*t)) + np.random.normal(0, 0.012))
        va_mf1  = min(0.93, 0.04 + 0.82 * (1 - np.exp(-4*t)) + np.random.normal(0, 0.018))
        tr_wf1  = tr_mf1 + np.random.normal(0, 0.008)
        va_wf1  = va_mf1 + np.random.normal(0, 0.010)

        # LR cosine-like decay
        lr = lr_base * (0.5 * (1 + np.cos(np.pi * t))) + 1e-6

        # Per-class F1 (val)
        base_pcf1 = [0.94, 0.82, 0.75, 0.70, 0.65, 0.72, 0.68, 0.60, 0.78, 0.74, 0.88]
        pcf1_v = [min(1.0, max(0.0, v * min(1.0, t * 1.6) + np.random.normal(0, 0.03)))
                  for v in base_pcf1]
        pcf1_t = [min(1.0, v + 0.04 + np.random.normal(0, 0.02)) for v in pcf1_v]

        if va_mf1 > best_f1:
            best_f1, best_ep = va_mf1, ep

        data['epochs'].append({
            'epoch': ep, 'lr': round(lr, 8), 'elapsed_s': round(40 + np.random.uniform(0, 5), 1),
            'train': {'loss': round(tr_loss, 5), 'accuracy': round(tr_acc, 5),
                      'macro_f1': round(tr_mf1, 5), 'weighted_f1': round(tr_wf1, 5),
                      'per_class_f1': [round(v, 4) for v in pcf1_t]},
            'val':   {'loss': round(va_loss, 5), 'accuracy': round(va_acc, 5),
                      'macro_f1': round(va_mf1, 5), 'weighted_f1': round(va_wf1, 5),
                      'per_class_f1': [round(v, 4) for v in pcf1_v]},
        })

    data['best_epoch']    = best_ep
    data['best_macro_f1'] = round(best_f1, 4)
    return data


# ============================================================
#  MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Plot hasil training ECG Holter')
    parser.add_argument('--log',  default=CNN_TRAINING_LOG,
                        help='Path ke training_log.json')
    parser.add_argument('--save', default='training_results.png',
                        help='Path output (default: training_results.png)')
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    if args.log:
        try:
            raw = load_log(args.log)
            print(f"✓ Loaded: {args.log}")
            is_demo = False
        except Exception as e:
            print(f"⚠ Gagal baca log: {e}\n  → Menggunakan data DEMO")
            raw = make_dummy_log()
            is_demo = True
    else:
        print("⚠ --log tidak diberikan → Menggunakan data DEMO")
        raw = make_dummy_log()
        is_demo = True

    d = extract(raw['epochs'])

    best_ep  = raw.get('best_epoch', int(d['epoch'][np.nanargmax(d['val_mf1'])]))
    best_f1  = raw.get('best_macro_f1', float(np.nanmax(d['val_mf1'])))
    cfg      = raw.get('config', {})
    model_type = cfg.get('model_type', 'ResNet-1D')
    n_epochs   = int(d['epoch'][-1])

    # ── Statistik ringkas ──────────────────────────────────────────────────
    print(f"\n  Model        : {model_type}")
    print(f"  Epochs       : {n_epochs}")
    print(f"  Best epoch   : {best_ep}")
    print(f"  Best val F1  : {best_f1:.4f}")
    best_idx = int(np.nanargmax(d['val_mf1']))
    print(f"  Best val Acc : {d['val_acc'][best_idx]*100:.2f}%")
    total_time = d['elapsed'].sum() / 3600
    print(f"  Total time   : {total_time:.1f} h")

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.42, wspace=0.30,
        left=0.06, right=0.97, top=0.90, bottom=0.06,
    )

    ax_loss  = fig.add_subplot(gs[0, 0])
    ax_mf1   = fig.add_subplot(gs[0, 1])
    ax_acc   = fig.add_subplot(gs[0, 2])
    ax_wf1   = fig.add_subplot(gs[1, 0])
    ax_lr    = fig.add_subplot(gs[1, 1])
    ax_pcf1  = fig.add_subplot(gs[1, 2])

    plot_loss(ax_loss, d, best_ep)
    plot_macro_f1(ax_mf1, d, best_ep)
    plot_accuracy(ax_acc, d, best_ep)
    plot_weighted_f1(ax_wf1, d, best_ep)
    plot_lr(ax_lr, d)
    plot_per_class_f1(ax_pcf1, d, best_ep)

    # ── Super title + info bar ─────────────────────────────────────────────
    demo_tag = '  ·  ⚠ DATA DEMO' if is_demo else ''
    fig.text(0.5, 0.965,
             f'Training Results  ·  {model_type}  ·  ECG Holter Arrhythmia Detection{demo_tag}',
             ha='center', va='top', fontsize=13.5, fontweight='bold', color='#1A1A2E')

    info = (f'Epochs: {n_epochs}   ·   Best epoch: {best_ep}   ·   '
            f'Best Val Macro-F1: {best_f1:.4f}   ·   '
            f'Best Val Acc: {d["val_acc"][best_idx]*100:.2f}%   ·   '
            f'Total training time: {total_time:.1f} h')
    fig.text(0.5, 0.944, info, ha='center', va='top', fontsize=9.5, color='#555')

    # ── Save ──────────────────────────────────────────────────────────────
    out = Path(args.save)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'\n✓ Saved → {out}')
    plt.show()


if __name__ == '__main__':
    main()