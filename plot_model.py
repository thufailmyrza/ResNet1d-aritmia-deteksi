"""
plot_model.py
==================
Visualisasi lengkap arsitektur ResNet-1D untuk ECG Holter Arrhythmia Detection.

Output (4 panel):
  1. Diagram arsitektur model (block diagram)
  2. Detail blok SE-ResBlock
  3. Pipeline data: Holter .bin → ONNX inference → arrhythmia.bin
  4. 11 kelas aritmia + flag bit-mapping

Cara pakai:
  python plot_model.py
  python plot_model.py --log checkpoints/cnn/training_log.json
  python plot_model.py --save output/architecture.png
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, ArrowStyle
from matplotlib.gridspec import GridSpec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_path import (

    CNN_TRAINING_LOG
)
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'figure.dpi': 150,
})

# ============================================================
#  WARNA PALETTE
# ============================================================
C_STEM   = '#4A90D9'
C_RES    = '#27AE60'
C_SE     = '#E67E22'
C_ATT    = '#9B59B6'
C_GAP    = '#1ABC9C'
C_HEAD   = '#E74C3C'
C_ARROW  = '#2C3E50'
C_BG     = '#F8F9FA'
C_BORDER = '#DEE2E6'
C_ONNX   = '#F39C12'
C_APP    = '#2980B9'
C_DARK   = '#1A1A2E'

ARRHYTHMIA_CLASSES = {
    0:  ('Normal',               '#95A5A6', 1),
    1:  ('Premature Beat',       '#3498DB', 2),
    2:  ('Bigeminy',             '#2ECC71', 4),
    3:  ('Trigeminy',            '#1ABC9C', 8),
    4:  ('Quadrigeminy',         '#27AE60', 16),
    5:  ('Couplet',              '#F39C12', 32),
    6:  ('Triplet',              '#E67E22', 64),
    7:  ('NSVT',                 '#E74C3C', 128),
    8:  ('Tachycardia',          '#C0392B', 256),
    9:  ('Bradycardia',          '#8E44AD', 512),
    10: ('Atrial Fibrillation',  '#2C3E50', 1024),
}


# ============================================================
#  HELPER: draw fancy box
# ============================================================
def draw_box(ax, x, y, w, h, label, sublabel='', color='#4A90D9',
             fontsize=8, alpha=0.92, radius=0.015):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor='white',
        linewidth=1.2, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    text_y = y + (0.008 if sublabel else 0)
    ax.text(x, text_y, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color='white', zorder=4, wrap=False)
    if sublabel:
        ax.text(x, y - 0.022, sublabel, ha='center', va='center',
                fontsize=fontsize - 1.5, color='#FFEAA7', zorder=4)


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.4, style='->', mutation=12):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f'->', color=color,
                                lw=lw, mutation_scale=mutation),
                zorder=5)


# ============================================================
#  PANEL 1: Arsitektur Model
# ============================================================
def draw_architecture(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(C_BG)
    ax.set_title('Arsitektur ResNet-1D  –  ECG Holter Arrhythmia Detector',
                 fontsize=11, fontweight='bold', pad=8, color=C_DARK)

    # Kolom tengah x
    cx = 0.5
    # Y posisi setiap layer (atas ke bawah)
    layers = [
        # (y,   label,                      sublabel,              color,  w,    h)
        (0.95,  'INPUT',                    '(B, 12, 2500)  float32 mV',  C_STEM,  0.35, 0.055),
        (0.855, 'STEM',                     'Conv1D k=15 s=2 → MaxPool\n(B, 64, 1250)',  C_STEM,  0.40, 0.065),
        (0.75,  'STAGE 1',                  '2× SE-ResBlock(64→64)  s=1\n(B, 64, 1250)', C_RES,   0.44, 0.065),
        (0.645, 'STAGE 2',                  '2× SE-ResBlock(64→128) s=2\n(B, 128, 625)', C_RES,   0.44, 0.065),
        (0.540, 'STAGE 3',                  '2× SE-ResBlock(128→256) s=2\n(B, 256, 313)',C_RES,   0.44, 0.065),
        (0.435, 'STAGE 4',                  '2× SE-ResBlock(256→512) s=2\n(B, 512, 157)',C_RES,   0.44, 0.065),
        (0.330, 'MULTI-SCALE ATT.',         'Temporal Attention 1D\n(B, 512, 157)', C_ATT,  0.42, 0.065),
        (0.225, 'GLOBAL AVG POOL',          '(B, 512)', C_GAP,   0.38, 0.055),
        (0.130, 'HEAD',                     'Linear 512→256 → BN → ReLU → Dropout\nLinear 256→11', C_HEAD, 0.46, 0.065),
        (0.030, 'OUTPUT  logits',           '(B, 11)  →  argmax  →  class index 0–10', '#C0392B', 0.46, 0.055),
    ]

    prev_y = None
    for (y, lbl, sub, color, w, h) in layers:
        draw_box(ax, cx, y, w, h, lbl, sub, color, fontsize=7.5)
        if prev_y is not None:
            arrow(ax, cx, prev_y - 0.035, cx, y + h/2 + 0.004)
        prev_y = y

    # Anotasi shortcut (skip connection)
    ax.annotate(
        'Residual\nshortcut', xy=(cx + 0.22 + 0.005, 0.680),
        xytext=(cx + 0.31, 0.590),
        arrowprops=dict(arrowstyle='->', color=C_SE, lw=1.2,
                        connectionstyle='arc3,rad=0.35'),
        fontsize=7, color=C_SE, fontweight='bold', zorder=6,
    )

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=C_STEM, label='Stem / Input'),
        mpatches.Patch(facecolor=C_RES,  label='SE-ResBlock'),
        mpatches.Patch(facecolor=C_ATT,  label='Temporal Attention'),
        mpatches.Patch(facecolor=C_GAP,  label='GAP'),
        mpatches.Patch(facecolor=C_HEAD, label='Head / Output'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=7,
              framealpha=0.85, ncol=1, borderpad=0.5)

    # Parameter count annotation
    ax.text(0.97, 0.97,
            'ResNet-1D "standard"\n~2.1 M params\nInput: 12-lead × 2500 samp\n(5 s @ 500 Hz)',
            ha='right', va='top', fontsize=7.5, color=C_DARK,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=C_BORDER, alpha=0.9))


# ============================================================
#  PANEL 2: Detail SE-ResBlock
# ============================================================
def draw_se_resblock(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(C_BG)
    ax.set_title('Detail: SE-ResBlock (Basic Block)', fontsize=10,
                 fontweight='bold', pad=6, color=C_DARK)

    cx = 0.42

    blocks = [
        (0.88, 'Input x',          '(B, C_in, L)',       C_STEM,  0.36, 0.058),
        (0.77, 'Conv1D  k=7',      'BN → ReLU',          C_RES,   0.38, 0.058),
        (0.66, 'Dropout',          '',                    '#7F8C8D',0.28, 0.048),
        (0.55, 'Conv1D  k=7',      'BN',                 C_RES,   0.38, 0.058),
        (0.42, 'SE Block',         'Squeeze → FC → FC\n→ Sigmoid × channel', C_SE, 0.44, 0.072),
        (0.28, 'Add + ReLU',       'F(x) + shortcut(x)', C_ATT,  0.38, 0.058),
        (0.16, 'Output',           '(B, C_out, L//stride)',C_GAP,  0.38, 0.058),
    ]

    prev_y = None
    for (y, lbl, sub, color, w, h) in blocks:
        draw_box(ax, cx, y, w, h, lbl, sub, color, fontsize=7.5)
        if prev_y is not None:
            arrow(ax, cx, prev_y - 0.03, cx, y + h/2 + 0.004)
        prev_y = y

    # Shortcut path
    sx = cx + 0.28
    ax.annotate('', xy=(sx, 0.305), xytext=(sx, 0.855),
                arrowprops=dict(arrowstyle='->', color=C_SE, lw=1.5,
                                connectionstyle='arc3,rad=0.0'))
    ax.plot([cx + 0.18, sx], [0.88, 0.88], color=C_SE, lw=1.4, ls='--')
    ax.plot([sx, cx + 0.19], [0.305, 0.305], color=C_SE, lw=1.4, ls='--')
    ax.text(sx + 0.02, 0.58, 'shortcut\n(Identity /\nConv1×1)',
            ha='left', va='center', fontsize=7, color=C_SE, fontweight='bold')

    # SE detail box
    se_x = 0.82
    se_y = 0.42
    box = FancyBboxPatch((se_x - 0.12, se_y - 0.14), 0.25, 0.28,
                         boxstyle='round,pad=0.01',
                         facecolor='#FFF3E0', edgecolor=C_SE, linewidth=1.5)
    ax.add_patch(box)
    ax.text(se_x, se_y + 0.10, 'SEBlock Detail', ha='center',
            fontsize=7.5, fontweight='bold', color=C_SE)
    for i, (step, desc) in enumerate([
        ('AdaptiveAvgPool', '(B,C,L)→(B,C,1)'),
        ('FC  C→C/r',       'ReLU'),
        ('FC  C/r→C',       'Sigmoid'),
        ('× input',         'channel scaling'),
    ]):
        yi = se_y + 0.055 - i * 0.055
        ax.text(se_x, yi, f'{step}  {desc}', ha='center',
                fontsize=6.5, color='#5D4037')


# ============================================================
#  PANEL 3: Pipeline Inference
# ============================================================
def draw_pipeline(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(C_BG)
    ax.set_title('Pipeline: Holter .bin  →  ONNX (CUDA)  →  arrhythmia.bin',
                 fontsize=10, fontweight='bold', pad=6, color=C_DARK)

    steps = [
        # (x, y, label, sub, color)
        (0.08, 0.70, 'Holter\n.bin', 'int16 raw ADC\n(n_samples × 12)', '#7F8C8D'),
        (0.24, 0.70, 'Convert\nmV', '× 0.0025\n→ float32', C_STEM),
        (0.40, 0.70, 'Window\nSliding', '2500 samp\n(stride ≥ 1)', C_RES),
        (0.56, 0.70, 'ONNX\nCUDA 13.1', 'CUDAExecution\nProvider', C_ONNX),
        (0.72, 0.70, 'ArgMax\n(B, 11)', 'class index\n0–10', C_ATT),
        (0.88, 0.70, 'arrhythmia\n.bin', '1<<class_idx\nint32/sample', C_HEAD),
    ]

    for (x, y, lbl, sub, color) in steps:
        draw_box(ax, x, y, 0.13, 0.16, lbl, sub, color, fontsize=7.5)

    for i in range(len(steps) - 1):
        x0 = steps[i][0] + 0.065
        x1 = steps[i+1][0] - 0.065
        y  = steps[i][1]
        arrow(ax, x0, y, x1, y, mutation=10)

    # ONNX Runtime box highlight
    x_onnx = steps[3][0]
    highlight = FancyBboxPatch((x_onnx - 0.09, 0.60), 0.18, 0.21,
                               boxstyle='round,pad=0.01',
                               facecolor='none', edgecolor=C_ONNX,
                               linewidth=2, linestyle='--', zorder=2)
    ax.add_patch(highlight)

    # Bottom: arrhythmia.bin detail
    ax.text(0.5, 0.47, 'arrhythmia.bin — Format per sample:', ha='center',
            fontsize=8.5, fontweight='bold', color=C_DARK)

    flag_x = [0.12, 0.30, 0.48, 0.66, 0.84]
    flag_samples = [
        ('Normal',    '2^0 = 1',   '#95A5A6'),
        ('Prem. Beat','2^1 = 2',   '#3498DB'),
        ('Bigeminy',  '2^2 = 4',   '#2ECC71'),
        ('Tachycardia','2^8 = 256','#E74C3C'),
        ('A-Fib',     '2^10= 1024','#2C3E50'),
    ]
    for x, (name, flag, color) in zip(flag_x, flag_samples):
        draw_box(ax, x, 0.32, 0.16, 0.10, name, flag, color, fontsize=7)

    ax.text(0.5, 0.16,
            'App parser: np.fromfile(arrhythmia.bin, dtype=np.int32)  →  '
            'int(np.log2(flag)) = class_index\n'
            'Class 0 (Normal, flag=1) dikecualikan parser — hanya class 1–10 '
            'ditampilkan sebagai region aritmia.',
            ha='center', va='center', fontsize=7.5, color='#4A4A4A',
            style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=C_BORDER))


# ============================================================
#  PANEL 4: 11 Kelas Aritmia
# ============================================================
def draw_classes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(C_BG)
    ax.set_title('11 Kelas Output Model  ·  Single-Label Classification',
                 fontsize=10, fontweight='bold', pad=6, color=C_DARK)

    cols = 2
    n = len(ARRHYTHMIA_CLASSES)
    rows = (n + cols - 1) // cols

    xs = [0.27, 0.76]
    ys = np.linspace(0.90, 0.08, rows)

    for idx, (cls_idx, (name, color, flag)) in enumerate(ARRHYTHMIA_CLASSES.items()):
        col = idx % cols
        row = idx // cols
        x = xs[col]
        y = ys[row]

        # Swatch
        sw = FancyBboxPatch((x - 0.22, y - 0.038), 0.44, 0.075,
                            boxstyle='round,pad=0.005',
                            facecolor=color, edgecolor='white',
                            linewidth=1, alpha=0.88, zorder=3)
        ax.add_patch(sw)

        ax.text(x - 0.18, y, f'[{cls_idx:2d}]', ha='left', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=4)
        ax.text(x - 0.06, y + 0.018, name, ha='left', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=4)
        ax.text(x - 0.06, y - 0.018, f'flag = 2^{cls_idx} = {flag}',
                ha='left', va='center', fontsize=7, color='#FFEAA7', zorder=4)

        if cls_idx == 0:
            ax.text(x + 0.19, y, '← dikecualikan\nparser (Normal)',
                    ha='left', va='center', fontsize=6.5,
                    color='#7F8C8D', style='italic')

    # Footer note
    ax.text(0.5, 0.02,
            'Output: logits (B, 11)  →  argmax  →  class index  →  1 << idx  →  int32 flag per ECG sample',
            ha='center', va='bottom', fontsize=7.5, color='#555',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=C_BORDER))


# ============================================================
#  PANEL 5 (opsional): Training curves dari training_log.json
# ============================================================
def draw_training_curves(ax_loss, ax_f1, log_path):
    """Plot loss & F1 per epoch dari training_log.json."""
    try:
        with open(log_path) as f:
            data = json.load(f)
        epochs_data = data.get('epochs', [])
        if not epochs_data:
            raise ValueError("Tidak ada data epoch")

        ep  = [d['epoch']         for d in epochs_data]
        trl = [d['train_loss']    for d in epochs_data]
        val = [d.get('val_loss',  [d.get('val_loss', None)]) for d in epochs_data]
        trf = [d.get('train_f1',  d.get('train_macro_f1', 0)) for d in epochs_data]
        vaf = [d.get('val_f1',    d.get('val_macro_f1',   0)) for d in epochs_data]

        # Loss
        ax_loss.plot(ep, trl, color=C_STEM, lw=2, label='Train Loss')
        if all(v is not None for v in val):
            ax_loss.plot(ep, val, color=C_HEAD, lw=2, ls='--', label='Val Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Cross-Entropy Loss')
        ax_loss.set_title('Training Loss', fontweight='bold')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_facecolor(C_BG)

        # F1
        ax_f1.plot(ep, trf, color=C_RES, lw=2, label='Train Macro-F1')
        ax_f1.plot(ep, vaf, color=C_ATT, lw=2, ls='--', label='Val Macro-F1')
        best_ep = ep[np.argmax(vaf)]
        best_f1 = max(vaf)
        ax_f1.axvline(best_ep, color='red', ls=':', lw=1.5,
                      label=f'Best ep={best_ep} F1={best_f1:.3f}')
        ax_f1.set_xlabel('Epoch')
        ax_f1.set_ylabel('Macro F1-Score')
        ax_f1.set_title('Macro F1-Score', fontweight='bold')
        ax_f1.legend()
        ax_f1.grid(True, alpha=0.3)
        ax_f1.set_facecolor(C_BG)

        return True
    except Exception as e:
        for ax in [ax_loss, ax_f1]:
            ax.text(0.5, 0.5, f'Training log tidak tersedia\n({e})',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray', style='italic')
            ax.axis('off')
        return False


# ============================================================
#  IMPROVED MODEL DIAGRAM
# ============================================================
def draw_improved_architecture(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    ax.set_facecolor(C_BG)
    ax.set_title('Varian: ImprovedResNet  [3, 4, 6, 3]  –  ResNet-50 style',
                 fontsize=10, fontweight='bold', pad=6, color=C_DARK)

    cx = 0.5
    stages_improved = [
        (0.94, 'INPUT',    '(B, 12, 2500)',           C_STEM, 0.35, 0.05),
        (0.86, 'STEM',     'Conv k=15 s=2 + MaxPool\n(B,64,1250)', C_STEM, 0.42, 0.06),
        (0.76, 'STAGE 1',  '3× Bottleneck(64→256)  s=1\n(B, 256, 1250)', C_RES,  0.46, 0.06),
        (0.66, 'STAGE 2',  '4× Bottleneck(256→512) s=2\n(B, 512, 625)',  C_RES,  0.46, 0.06),
        (0.56, 'STAGE 3',  '6× Bottleneck(512→1024) s=2\n(B,1024, 313)', C_RES,  0.46, 0.06),
        (0.46, 'STAGE 4',  '3× Bottleneck(1024→2048) s=2\n(B,2048, 157)',C_RES,  0.46, 0.06),
        (0.36, 'MULTI-SCALE ATT.', 'Temporal Attention 1D\n(B,2048,157)',C_ATT,  0.44, 0.06),
        (0.26, 'GAP',      '(B, 2048)',               C_GAP,  0.36, 0.05),
        (0.16, 'HEAD',     'Linear 2048→512→11\nBN + Dropout',    C_HEAD, 0.44, 0.06),
        (0.06, 'OUTPUT',   '(B, 11) logits',          '#C0392B',0.40, 0.05),
    ]

    prev_y = None
    for (y, lbl, sub, color, w, h) in stages_improved:
        draw_box(ax, cx, y, w, h, lbl, sub, color, fontsize=7.2)
        if prev_y is not None:
            arrow(ax, cx, prev_y - 0.03, cx, y + h/2 + 0.004)
        prev_y = y

    ax.text(0.02, 0.5,
            'Bottleneck\n1×1 → 3×3 → 1×1\n+ SE\n+ Dropout',
            ha='left', va='center', fontsize=7, color=C_SE,
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor=C_SE))

    ax.text(0.97, 0.5,
            'ImprovedResNet\n~15 M params\n(cocok untuk\ndataset besar)',
            ha='right', va='center', fontsize=7.5, color=C_DARK,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=C_BORDER))


# ============================================================
#  MAIN PLOT
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Plot arsitektur ResNet-1D ECG Holter Arrhythmia Detector')
    parser.add_argument('--log',  default=CNN_TRAINING_LOG,
                        help='Path ke training_log.json (opsional)')
    parser.add_argument('--save', default='architecture_holter_ecg.png',
                        help='Path output gambar (default: architecture_holter_ecg.png)')
    args = parser.parse_args()

    has_log = args.log and Path(args.log).exists()

    # ── Figure layout ──────────────────────────────────────────────────────
    if has_log:
        fig = plt.figure(figsize=(26, 22), facecolor='white')
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30,
                      left=0.04, right=0.97, top=0.95, bottom=0.04)
        ax1 = fig.add_subplot(gs[0, 0])   # arsitektur standard
        ax2 = fig.add_subplot(gs[0, 1])   # SE-ResBlock detail
        ax3 = fig.add_subplot(gs[0, 2])   # pipeline
        ax4 = fig.add_subplot(gs[1, 0])   # 11 kelas
        ax5 = fig.add_subplot(gs[1, 1])   # improved arch
        ax6 = fig.add_subplot(gs[1, 2])   # training loss
        ax7 = fig.add_subplot(gs[2, 0:2]) # training F1 (span 2 cols)
        extra_axes = [ax6, ax7]
        draw_training_curves(ax6, ax7, args.log)
    else:
        fig = plt.figure(figsize=(26, 16), facecolor='white')
        gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28,
                      left=0.04, right=0.97, top=0.95, bottom=0.04)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])

    draw_architecture(ax1)
    draw_se_resblock(ax2)
    draw_pipeline(ax3)
    draw_classes(ax4)
    draw_improved_architecture(ax5)

    # ── Super title ────────────────────────────────────────────────────────
    fig.text(0.5, 0.985,
             'ResNet-1D  ·  ECG Holter Arrhythmia Detection  ·  12-lead · 11 kelas · ONNX CUDA 13.1',
             ha='center', va='top', fontsize=13, fontweight='bold',
             color=C_DARK)
    fig.text(0.5, 0.970,
             'Input: (B, 12, 2500) float32 mV   ·   Output: (B, 11) logits   ·   '
             'Export: .onnx + .pkl   ·   Runtime: CUDAExecutionProvider',
             ha='center', va='top', fontsize=9, color='#555')

    out = Path(args.save)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'\n✓ Saved → {out}')
    plt.show()


if __name__ == '__main__':
    main()