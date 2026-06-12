# Deployment & Integration

This document describes how a trained checkpoint becomes a deployable ONNX model, the exact contract expected by the Holter application's `ArrhythmiaDetector`, validation results against real device recordings, and the fixes required on the application side to close the train/inference domain gap.

## Contents

- [Export Pipeline](#export-pipeline)
- [ONNX I/O Contract](#onnx-io-contract)
- [ONNX Runtime Setup](#onnx-runtime-setup)
- [`arrhythmia.bin` Format](#arrhythmiabin-format)
- [Integration Steps](#integration-steps)
- [Validation on Target Hardware](#validation-on-target-hardware)
- [Root Cause Analysis & Required Fixes](#root-cause-analysis--required-fixes)
- [Hardware Domain Gap (Leads III / aVF)](#hardware-domain-gap-leads-iii--avf)
- [Packaging Notes (PyInstaller)](#packaging-notes-pyinstaller)
- [Deployment Checklist](#deployment-checklist)

---

## Export Pipeline

`train/export_model.py` provides three entrypoints:

```bash
# Export ONNX + PKL from a checkpoint
python train/export_model.py export --checkpoint OUTPUT/checkpoints/cnn/best_model.pth \
    --output-dir OUTPUT/exported_models --model-type standard

# Run inference on a single recording → arrhythmia.bin
python train/export_model.py infer --model OUTPUT/exported_models/arrhythmia_model.onnx \
    --input path/to/recording.bin --source device

# Batch inference over a folder of recordings
python train/export_model.py batch --model OUTPUT/exported_models/arrhythmia_model.onnx \
    --input path/to/recordings_dir --source device
```

- `export_to_onnx()` — loads the checkpoint, builds the model via `build_model()`, traces with a dummy `(1, 12, 2500)` input, and exports with `input_names=['ecg_input']`, `output_names=['arrhythmia_logits']`, dynamic batch axis, `opset_version=14`. It then verifies the exported graph with `onnxruntime` (CPU) and prints the predicted class for the dummy input.
- `export_to_pkl()` — wraps the model in `HolterArrhythmiaModel` (a thin inference wrapper exposing `predict_class`, `predict_flags`, `infer_recording`) and pickles it, alongside a `model_metadata.json` describing the I/O contract, class labels, ADC gains, and `arrhythmia.bin` encoding.
- `run_inference()` / `run_inference_batch()` — convert a raw `.bin` recording to `arrhythmia.bin` using either the `.pkl` or `.onnx` model.

> **Known issue**: the `export` subcommand's `--model-type` choices are `['standard', 'improved']` — `resnet152` is not listed, even though `build_model()` supports it. To export the `resnet152` checkpoint, either call `export_pipeline(..., model_type='resnet152')` directly in a script, or use `tools/fix_export.py` (see below), which hardcodes `MODEL_TYPE = "resnet152"`.

### Single-file ONNX export (`tools/fix_export.py`)

Larger checkpoints (e.g. `resnet152`, ~25M params) can export as a split `.onnx` + `.onnx.data` pair (ONNX "external data" format) once the model exceeds the 2 GB protobuf limit threshold in some exporters. `tools/fix_export.py`:

1. Exports without external data parameters first.
2. If an `.onnx.data` file is still produced, loads the model, calls `load_external_data_for_model()` to pull the weights back in, and re-saves with `onnx.save(..., save_as_external_data=False)`.
3. Deletes the now-unused `.onnx.data` file and runs a quick `onnxruntime` verification.

The result is a **single-file `.onnx`** suitable for copying directly into the application's `resource/models/` folder.

---

## ONNX I/O Contract

| | Name | Shape | Dtype |
|---|---|---|---|
| Input | `ecg_input` | `(batch, 12, 2500)` | `float32`, millivolts |
| Output | `arrhythmia_logits` | `(batch, 11)` | `float32`, raw logits |

The application is responsible for:
1. Converting raw device samples to millivolts.
2. Applying the same bandpass filter used during training (0.5–40 Hz Butterworth) — **see [Root Cause Analysis](#root-cause-analysis--required-fixes)**.
3. Splitting the recording into non-overlapping 2500-sample windows (padding the final window with zeros if needed).
4. Running `argmax` on `arrhythmia_logits` to get `class_index` per window.
5. Expanding `class_index` → `1 << class_index` and repeating across all 2500 samples of the window to build the per-sample `arrhythmia.bin` stream.

Steps 3–5 are already implemented in `ArrhythmiaDetector.predict()` (`arrhythmia_detector.py`) and `_onnx_infer_recording()` (`train/export_model.py`) — they should remain in sync if either is modified.

---

## ONNX Runtime Setup

The application requests `providers=['CUDAExecutionProvider', 'CPUExecutionProvider']` — `onnxruntime` falls back to CPU automatically if CUDA initialization fails, so a missing/incompatible GPU does not crash detection, only slows it down.

Before relying on GPU inference in production:

1. Confirm CUDA Toolkit visibility with `tools/diagnostics/verify_cuda.py` (`torch.cuda.is_available()`, `torch.cuda.get_device_name(0)`).
2. Confirm `onnxruntime-gpu` actually picks up `CUDAExecutionProvider` for your exported model with `tools/verify_onnx.py` — it prints input/output node shapes, runs a dummy inference, and reports the predicted class and a sanity cross-entropy loss (`≈ ln(11) ≈ 2.40` for an untrained/random model).
3. **Check the `onnxruntime-gpu` ↔ CUDA Toolkit compatibility matrix** for your target CUDA version before deployment. `onnxruntime-gpu` builds are tied to specific CUDA major versions (e.g. CUDA 12.x); a CUDA 13.x host may require a newer `onnxruntime-gpu` release than what's pinned in `requirements.txt`. If `CUDAExecutionProvider` fails to initialize, `onnxruntime` will silently fall back to CPU — verify which provider is actually active rather than assuming.

---

## `arrhythmia.bin` Format

- **One `int32` per ECG sample**, same length as the input recording.
- `value = 1 << class_index` (i.e. `1, 2, 4, 8, ..., 1024` for classes 0–10).
- Parsed by `arrhythmia_parser.py`:
  ```python
  raw = np.fromfile("arrhythmia.bin", dtype=np.int32)
  work = np.where(raw > 0, np.log2(raw).astype(np.int32), 0)   # class_index per sample
  # work == 0  → Normal, excluded from arrhythmia overlay
  # work 1–10  → contiguous runs become arrhythmia "regions" with (start, end, flag)
  ```
- `find_arrhythmia_regions()` collapses contiguous same-class runs into `(start_sample, end_sample, flag)` records, which the viewer (`ecg_viewer.py`, `ecg_print_dialog.py`) renders as annotated regions on the ECG trace.

---

## Integration Steps

1. Export the trained checkpoint to a single-file `.onnx` (see above).
2. Copy it to `resource/models/detector.onnx` in the Holter application (path resolved via `utils.get_resource_path()`, which handles both source and PyInstaller-frozen execution).
3. In `controls_panel.py`, the "Run Arrhythmia Detector" button is gated behind a check — ensure `self.run_detector_btn.setEnabled(True)` / `setText("Run Arrhythmia Detector")` is active when no `arr.bin` exists yet for the loaded record (see `update_arrhythmia_data()`).
4. Run a recording through the detector (`ECGViewer.run_arrhythmia_detection()`), which writes `arr.bin` next to the recording and reloads `arrhythmia_data` via `find_arrhythmia_regions()`.
5. Apply the pipeline fixes below **before** trusting detection results on real device recordings.

---

## Validation on Target Hardware

The exported `resnet152` model (epoch 63, Macro F1 = 0.8499 on the PTB-XL+INCART test split) was evaluated against **6 Xirka Smart Holter device recordings**:

| Result | Count | Notes |
|---|---|---|
| Pass | 2 | Predictions consistent with expected annotations |
| Partial | 2 | Bigeminy / Trigeminy partially correct — some windows misclassified within the same episode |
| Fail | 4* | See misclassification patterns below |

\* counts reflect the 6-recording validation set used during development; treat as a snapshot, not a guaranteed pass rate — re-run after applying the fixes below.

**Observed misclassification patterns:**

- **Normal → Premature Beat** (false positives on normal segments)
- **Tachycardia ↔ Atrial Fibrillation** (bidirectional confusion)
- **Atrial Fibrillation → Bradycardia**

These patterns are consistent with a **systematic amplitude/frequency-content mismatch** between the device input the model receives at inference time and the data it was trained on — not a model capacity problem. See the next section.

---

## Root Cause Analysis & Required Fixes

### 1. ADC gain mismatch

The model was trained on signals scaled by `ADC_GAIN_DEVICE = 0.0025` (`config_path.py`) — i.e. `mV = raw_int16 × 0.0025`. The application's `ECGViewer` currently converts raw samples to mV using `self.adc_to_mv_factor` (defined in `ecg_viewer.py`'s `__init__`), and this **same factor** is reused both for on-screen plotting (`update_plot_data`, `_compute_lead_layout`) and for the detector's input in `run_arrhythmia_detection()`:

```python
ecg_mv = raw_struct['leads'].astype(np.float32) * self.adc_to_mv_factor
flags = detector.predict(ecg_mv, ...)
```

If `self.adc_to_mv_factor != ADC_GAIN_DEVICE (0.0025)`, the model receives signals at the wrong amplitude — directly producing the misclassification patterns above (an amplitude-shifted signal looks like a different rhythm to an amplitude-sensitive model).

**Fix options:**

- **Preferred**: introduce a dedicated constant for the detector input, decoupled from the display gain, e.g.:
  ```python
  # config_path.py already defines this — import and reuse it:
  from config_path import ADC_GAIN_DEVICE   # = 0.0025

  # in run_arrhythmia_detection():
  ecg_mv = raw_struct['leads'].astype(np.float32) * ADC_GAIN_DEVICE
  ```
  This avoids changing the on-screen trace amplitude (`self.adc_to_mv_factor`), which may be intentionally tuned for display readability and is independent of what the model expects.
- **Alternative**: if `self.adc_to_mv_factor` is *also* intended to represent the true ADC-to-mV conversion (and the display should reflect calibrated mV), set it to `0.0025` and verify the on-screen trace amplitude is still clinically reasonable (compare against a known-good reference recording).

Either way, **verify with `tools/diagnostics/probe_xirka.py`**, which prints per-lead mV ranges and standard deviations for raw `.bin` files using `ADC_TO_MV = 0.0025` — compare these ranges against what the training pipeline produces for PTB-XL/INCART windows (typically sub-millivolt to a few mV per lead).

### 2. Missing bandpass filter at inference time

`preprocess_ptbxl.py` and `convert_incart.py` both apply a **0.5–40 Hz, 4th-order Butterworth bandpass filter** (`filtfilt`, zero-phase) before windowing — every training window has had baseline wander and high-frequency noise removed.

`arrhythmia_detector.py`'s `predict()` does **not** apply any filtering — it reshapes the raw (gain-converted) signal directly into windows and feeds them to the ONNX model. This is a second, independent source of train/inference mismatch: unfiltered baseline wander and powerline/EMG noise present in real device recordings are out-of-distribution for the model.

**Fix**: apply the same filter before windowing, e.g. in `arrhythmia_detector.py`:

```python
from scipy.signal import butter, filtfilt

def _bandpass_filter(signal, fs=500, lowcut=0.5, highcut=40.0, order=4):
    """4th-order Butterworth bandpass, zero-phase — matches preprocess_ptbxl.bandpass_filter()."""
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)

# inside ArrhythmiaDetector.predict(), before padding/windowing:
ecg_signal_mv = self._bandpass_filter(ecg_signal_mv)
```

Apply the filter to the **whole recording** (or sufficiently long chunks) before windowing — `filtfilt` on very short windows (2500 samples) can introduce edge artifacts; filtering the full signal first and then slicing windows matches how the training data was produced (filter-then-window, not window-then-filter).

### 3. Hardware domain gap (see next section)

---

## Hardware Domain Gap (Leads III / aVF)

Per-lead amplitude analysis (`tools/diagnostics/probe_xirka.py`) shows that **leads III and aVF** on the Xirka device — which uses a torso (not limb) electrode placement — are systematically **lower amplitude** than the corresponding leads in PTB-XL/INCART.

This is treated as an **accepted hardware limitation** rather than a pipeline bug: it cannot be fixed by recalibrating gain or filtering alone, since it reflects a genuine difference in electrode geometry. It is **partially mitigated** by:

- The existing `±30%` amplitude-scaling augmentation (`HolterECGDataset._augment`, see [TRAINING.md](TRAINING.md)), which exposes the model to a range of per-recording amplitude scales during training.
- The recommended **fine-tuning pass on real Xirka device recordings** (see [TRAINING.md — Fine-Tuning for Target Hardware](TRAINING.md#fine-tuning-for-target-hardware)), which adapts the classification head to the device's actual lead-III/aVF amplitude distribution without requiring a full retrain.

If lead III / aVF amplitude remains a dominant error source after fixes 1–2 above and fine-tuning, consider lead-specific gain calibration as a device-side firmware/preprocessing change (outside the scope of this repository).

---

## Packaging Notes (PyInstaller)

When building the application with PyInstaller (see the application repository's README), the ONNX-enabled build requires:

- `--hidden-import="onnxruntime"`
- `--add-data "resource/models;resource/models"` (bundles `detector.onnx`)
- Explicit DLL bundling for `onnxruntime.dll` and `onnxruntime_providers_shared.dll`, with `os.add_dll_directory()` called early in a frozen context (`getattr(sys, 'frozen', False)`) so `onnxruntime` can locate its native dependencies at runtime.
- The "without model" build variant excludes `onnxruntime` and `arrhythmia_detector` entirely (`--exclude-module`) for a smaller binary when arrhythmia detection is not needed.

---

## Deployment Checklist

- [ ] Export checkpoint to a **single-file** `.onnx` (`tools/fix_export.py` if needed)
- [ ] Verify `ecg_input` / `arrhythmia_logits` shapes and a sanity inference with `tools/verify_onnx.py`
- [ ] Confirm `CUDAExecutionProvider` actually loads for your target CUDA version, or accept CPU fallback
- [ ] Align the detector's ADC-to-mV conversion with `ADC_GAIN_DEVICE = 0.0025` (decoupled from display gain)
- [ ] Add the 0.5–40 Hz bandpass filter to `ArrhythmiaDetector.predict()`, applied before windowing
- [ ] Re-run the 6-recording validation set (or equivalent) and compare against the misclassification patterns above
- [ ] (Recommended) Fine-tune on real device recordings per [TRAINING.md](TRAINING.md)
- [ ] Copy the final `.onnx` to `resource/models/detector.onnx` and rebuild the PyInstaller bundle
