# Architecture

This document describes the model architectures implemented in `model/resnet1d.py` and `model/resnet152.py`, the shared I/O contract, and the end-to-end inference pipeline that connects the trained model to the Holter application.

## Contents

- [Shared I/O Contract](#shared-io-contract)
- [Variant: `standard` (SE-ResNet-1D)](#variant-standard-se-resnet-1d)
- [Variant: `improved`](#variant-improved)
- [Variant: `resnet152` (Bottleneck ResNet-1D)](#variant-resnet152-bottleneck-resnet-1d)
- [Building Blocks](#building-blocks)
- [Inference Pipeline & `arrhythmia.bin`](#inference-pipeline--arrhythmiabin)
- [Choosing a Variant](#choosing-a-variant)

---

## Shared I/O Contract

All three model variants implement the same interface (see `build_model()` in `model/resnet1d.py`):

```python
model = build_model(model_type, num_classes=11, num_channels=12, dropout=0.3)

logits = model(x)                # x: (B, 12, 2500) float32, mV  →  (B, 11) logits
cls    = model.predict_class(x)  # (B,) int64, class index 0–10, always runs in eval mode
flags  = model.predict_flag(x)   # (B,) int32, = 1 << class_index
```

- **Input**: `(batch, 12, 2500)` — 12-lead ECG, 5-second window at 500 Hz, scale in **millivolts** (not normalized to `[-1, 1]`).
- **Output**: `(batch, 11)` raw logits (no softmax) — use `CrossEntropyLoss` for training, `argmax` for inference.
- **Lead order**: `I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6` (see `ECG_CHANNELS` in `config_path.py`).
- `predict_class()` and `predict_flag()` temporarily switch the model to `eval()` (disabling dropout) regardless of the caller's current mode, then restore the original mode — this guarantees deterministic, mutually consistent outputs even if called during training.

---

## Variant: `standard` (SE-ResNet-1D)

~2.1M parameters. Default variant, defined by `ResNet1D` with `layers=[2, 2, 2, 2]`.

```
Input (B, 12, 2500)
   │
   ▼
Stem
  Conv1D(in=12, out=64, k=15, s=2, p=7) → BatchNorm1D → ReLU → MaxPool1D(k=3, s=1, p=1)
  → (B, 64, 1250)
   │
   ▼
Stage 1 — 2× SE-ResBlock(64 → 64,   stride=1)   → (B, 64,  1250)
Stage 2 — 2× SE-ResBlock(64 → 128,  stride=2)   → (B, 128, 625)
Stage 3 — 2× SE-ResBlock(128 → 256, stride=2)   → (B, 256, 313)
Stage 4 — 2× SE-ResBlock(256 → 512, stride=2)   → (B, 512, 157)
   │
   ▼
Multi-Scale Temporal Attention (optional, default ON) → (B, 512, 157)
   │
   ▼
Global Average Pool (AdaptiveAvgPool1d) → (B, 512)
   │
   ▼
Head
  Linear(512 → 256) → BatchNorm1D → ReLU → Dropout(p) → Linear(256 → 11)
   │
   ▼
Output (B, 11) logits
```

Channel progression: `64 → 128 → 256 → 512`, doubling at each stage transition (stride 2), while the temporal length shrinks from 2500 → 1250 → 625 → 313 → 157.

---

## Variant: `improved`

Same building blocks as `standard`, but with a deeper stage layout `[3, 4, 6, 3]` (ResNet-50-style depth), giving ~15M parameters. Recommended for larger datasets where the standard variant underfits. Implemented as `ImprovedResNet(ResNet1D)` — only the `layers` argument differs.

---

## Variant: `resnet152` (Bottleneck ResNet-1D)

~25M parameters. **This is the currently deployed variant** (best checkpoint: epoch 63, Macro F1 = 0.8499). Implemented in `model/resnet152.py` as `ResNet152ECG`, ported from a Keras/TensorFlow notebook to PyTorch.

Differences from `standard` / `improved`:

| | `standard` / `improved` | `resnet152` |
|---|---|---|
| Block type | SE-ResBlock (2-layer, "wide") | Bottleneck (1×1 → 3×3 → 1×1, "deep") |
| Temporal attention | Multi-scale attention after stage 4 | Not used |
| Stem kernel | 15 | 7 |
| Stage layout | `[2,2,2,2]` or `[3,4,6,3]` | `[3,4,6,3]` Bottleneck blocks (fixed) |
| Head | `Linear(512 → 256 → 11)` | `Linear(2048 → 256 → 128 → 11)` |
| Weight init | PyTorch default (Kaiming for convs) | Xavier Uniform (Glorot), matching the original Keras notebook |

```
Input (B, 12, 2500)
   │
   ▼
Stem
  ZeroPad1D(3) → Conv1D(12 → 64, k=7, s=2) → BN → ReLU → MaxPool1D(k=3, s=2, p=1)
   │
   ▼
Stage 2 (64 → 256):  ConvBlock([128,128,256], s=1) + 2× IdentityBlock
Stage 3 (256 → 512): ConvBlock([128,128,512], s=2) + 3× IdentityBlock
Stage 4 (512 → 1024): ConvBlock([256,256,1024], s=2) + 5× IdentityBlock
Stage 5 (1024 → 2048): ConvBlock([512,512,2048], s=2) + 2× IdentityBlock
   │
   ▼
Global Average Pool → (B, 2048)
   │
   ▼
Head
  Dropout(p=0.5) → Linear(2048→256) → ReLU → Linear(256→128) → ReLU → Linear(128→11)
   │
   ▼
Output (B, 11) logits
```

Each `ConvolutionalBlock1D` projects the shortcut path with a `1×1` convolution (and stride `s`) to match the main path's output channels/length; each `IdentityBlock1D` requires `in_channels == F3` and uses a plain identity shortcut.

---

## Building Blocks

### SEBlock (Squeeze-and-Excitation)

Used inside every `ResBlock1D` (standard/improved variants). Recalibrates channel-wise importance — e.g. giving more weight to leads that are more informative for a given arrhythmia pattern.

```
x: (B, C, L)
  → AdaptiveAvgPool1d(1)        → (B, C)
  → Linear(C → C/reduction) → ReLU
  → Linear(C/reduction → C) → Sigmoid   → (B, C)
  → x * scale (broadcast over L)
```

Default `reduction=16`, with a minimum bottleneck width of 4 channels.

### ResBlock1D (SE-ResBlock)

```
identity = shortcut(x)                 # Identity, or Conv1x1+BN if shape changes
out = ReLU(BN(Conv1D_k7(x)))
out = Dropout(out)
out = BN(Conv1D_k7(out))
out = SEBlock(out)
return ReLU(out + identity)
```

### BottleneckBlock1D / ConvolutionalBlock1D / IdentityBlock1D

Used by `resnet152`. `ConvolutionalBlock1D` projects the shortcut (`Conv1x1` with stride `s` + BN); `IdentityBlock1D` requires no shape change and adds the input directly. Internally each block is `1×1 → k×1 → 1×1` with BatchNorm and ReLU between the first two convolutions.

### MultiScaleAttention (temporal)

Applied once, after the final residual stage (standard/improved only):

```
x: (B, C, L)
avg = mean(x, dim=channels)   → (B, 1, L)
mx  = max(x,  dim=channels)   → (B, 1, L)
combined = concat([avg, mx])  → (B, 2, L)
attn = Sigmoid(Conv1D_k7(combined))  → (B, 1, L)
return x * attn
```

This lets the model down-weight time segments that are uninformative (e.g. a window that only partially overlaps an arrhythmic episode).

---

## Inference Pipeline & `arrhythmia.bin`

The full deployment pipeline, from a raw Holter `.bin` recording to the application's arrhythmia overlay, is:

```
Holter .bin (int16 raw ADC, n_samples × 12)
   │  × ADC_GAIN_DEVICE (0.0025)  →  mV
   ▼
Sliding window: 2500 samples (5 s), non-overlapping at inference time
   │
   ▼
ONNX model (CUDAExecutionProvider / CPUExecutionProvider)
   │  forward → (B, 11) logits
   ▼
argmax → class_index per window (0–10)
   │
   ▼
flag = 1 << class_index → int32, repeated for every sample in the window
   │
   ▼
arrhythmia.bin (one int32 per ECG sample)
```

**Parsing in the application** (`arrhythmia_parser.py`):

```python
data = np.fromfile("arrhythmia.bin", dtype=np.int32)
class_index = np.log2(flag)     # only for flag > 0
# class_index == 0  → Normal, excluded from the arrhythmia overlay
# class_index 1–10  → rendered as an arrhythmia region
```

For full details on the export step that produces the `.onnx` file and the exact contract expected by `ArrhythmiaDetector`, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Choosing a Variant

| Variant | Params | When to use |
|---|---|---|
| `standard` | ~2.1M | Fast iteration, CPU-only debugging, smaller deployment footprint |
| `improved` | ~15M | More capacity than `standard` while keeping the SE-ResBlock + temporal attention design |
| `resnet152` | ~25M | Best observed accuracy on the merged PTB-XL + INCART dataset; currently the deployed model |

All variants are interchangeable at the I/O level — switching variants only requires re-training (or fine-tuning) and re-exporting to ONNX; the application-side contract does not change.
