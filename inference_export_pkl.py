"""
inference_export_pkl.py  –  REVISI v3
Export model terlatih ke ONNX + .pkl, dan jalankan inference untuk menghasilkan
arrhythmia.bin yang kompatibel dengan aplikasi Holter ECG.

============================================================
KONTRAK DENGAN APP (arrhythmia_detector.py + arrhythmia_parser.py):
------------------------------------------------------------
  arrhythmia_detector.py:
    - Input : ecg_signal_mv (n_samples, 12) float32, SKALA mV, tanpa normalisasi
    - Reshape: → (batch, 12, 2500) float32
    - Inference: model ONNX → logits (batch, 11)
    - Prediksi: np.argmax(logits, axis=1) → class_index per window
    - Flag    : 1 << class_index → int32 per sample
    - Output  : arrhythmia.bin = per-sample int32 array

  arrhythmia_parser.py:
    - Baca: np.fromfile(path, dtype=np.int32) → (n_samples,)
    - Decode: np.log2(flag) → class_index (0 = normal, dikecualikan)
    - Region aritmia: class_index 1–10 (flag 2–1024)
============================================================

Alur export:
  1. Load checkpoint PyTorch (.pth)
  2. Trace model → ONNX (dipakai langsung oleh app)
  3. Bungkus model → .pkl (untuk inference custom / testing)
  4. Inference test: .bin Holter → arrhythmia.bin per-sample int32
"""

import torch
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_path import (
    CNN_BEST_MODEL, EXPORTED_MODELS_DIR, ONNX_MODEL_PATH, PKL_MODEL_PATH,
    ARRHYTHMIA_CLASSES, ARRHYTHMIA_LABELS, NUM_ARRHYTHMIA_CLASSES,
    NUM_CHANNELS, HOLTER_SAMPLING_RATE,
    ADC_GAIN_DEVICE, ADC_GAIN_INT16, INT16_TO_MV,
    WINDOW_SIZE,
)
from resnet1d import build_model, ResNet1D

# Faktor konversi raw ADC Holter → mV
# App menerima mV, jadi sebelum feed ke model:
#   raw_adc × ADC_GAIN_DEVICE = mV
RAW_ADC_TO_MV = ADC_GAIN_DEVICE   # = 0.0025


# ============================================================================
# MODEL WRAPPER (.pkl)
# ============================================================================

class HolterArrhythmiaModel:
    """
    Wrapper model PyTorch untuk serialisasi .pkl dan inference.
    Digunakan untuk testing / integrasi non-ONNX.

    Kontrak output:
      predict_class(x)    → class index (0–10) per window
      predict_flags(x)    → (1 << class_index) int32 per window
      infer_recording(ecg_mv) → per-sample int32 array (= arrhythmia.bin content)
    """

    def __init__(self, model: ResNet1D, device: str = 'cpu'):
        self.model         = model.to(device).eval()
        self.device        = device
        self.class_labels  = ARRHYTHMIA_LABELS
        self.num_classes   = NUM_ARRHYTHMIA_CLASSES
        self.window_size   = WINDOW_SIZE

    # ── Core Inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: (B, 12, W) float32 mV

        Returns:
            (B, 11) float32 logits
        """
        return self.model(x.to(self.device)).cpu().numpy()

    @torch.no_grad()
    def predict_class(self, x: torch.Tensor) -> np.ndarray:
        """Return class index 0–10 per window. Shape: (B,)."""
        logits = self.predict_logits(x)
        return np.argmax(logits, axis=1).astype(np.int32)

    @torch.no_grad()
    def predict_flags(self, x: torch.Tensor) -> np.ndarray:
        """
        Return flag values (1 << class_index) per window.
        Shape: (B,) int32. Cocok dengan output app ArrhythmiaDetector.
        """
        cls = self.predict_class(x)
        return (1 << cls).astype(np.int32)

    # ── Inference Rekaman Lengkap ─────────────────────────────────────────────

    def infer_recording(self, ecg_mv: np.ndarray,
                         batch_size: int = 32) -> np.ndarray:
        """
        Inference pada seluruh rekaman ECG → per-sample int32 flags.

        Mengikuti logika ArrhythmiaDetector.predict() di app:
          1. Pad agar n_samples % window_size == 0
          2. Potong menjadi window-window non-overlapping
          3. Argmax per window → flag per window
          4. Repeat flag ke setiap sample dalam window
          5. Hilangkan padding

        Args:
            ecg_mv  : (n_samples, 12) atau (12, n_samples) float32, satuan mV
            batch_size: Ukuran batch inference

        Returns:
            flags: (n_samples,) int32
              • Normal windows: 1  (2^0, dikecualikan oleh parser)
              • Aritmia windows: 2–1024  (2^1–2^10)
        """
        # Pastikan shape (n_samples, 12)
        if ecg_mv.ndim == 2 and ecg_mv.shape[0] == NUM_CHANNELS:
            ecg_mv = ecg_mv.T
        n_samples, n_leads = ecg_mv.shape
        assert n_leads == NUM_CHANNELS, f"Butuh {NUM_CHANNELS} lead, dapat {n_leads}"

        # Padding agar kelipatan window_size
        remainder = n_samples % self.window_size
        pad_len   = (self.window_size - remainder) % self.window_size
        if pad_len > 0:
            padded = np.pad(ecg_mv, ((0, pad_len), (0, 0)), mode='constant')
        else:
            padded = ecg_mv

        total_windows   = padded.shape[0] // self.window_size
        all_predictions = []

        for i in range(0, total_windows, batch_size):
            end  = min(i + batch_size, total_windows)
            bsz  = end - i
            chunk = padded[i*self.window_size : end*self.window_size]
            # (B, W, 12) → transpose → (B, 12, W)
            model_input = (chunk.reshape(bsz, self.window_size, NUM_CHANNELS)
                                .transpose(0, 2, 1)
                                .astype(np.float32))
            x   = torch.from_numpy(model_input)
            cls = self.predict_class(x)
            all_predictions.append(cls)

        window_labels = np.concatenate(all_predictions)            # (total_windows,)
        sample_labels = np.repeat(window_labels, self.window_size) # (total_padded,)

        if pad_len > 0:
            sample_labels = sample_labels[:-pad_len]

        flags = (1 << sample_labels).astype(np.int32)
        return flags   # (n_samples,) – siap ditulis ke arrhythmia.bin


# ============================================================================
# EXPORT ONNX
# ============================================================================

def export_to_onnx(checkpoint_path=None, output_path=None,
                   model_type: str = 'standard') -> Path:
    """
    Export model PyTorch → ONNX untuk dipakai ArrhythmiaDetector di app.

    Input node : 'ecg_input'   shape (batch, 12, 2500) float32 mV
    Output node: 'arrhythmia_logits'  shape (batch, 11) float32

    Returns:
        Path ke file ONNX
    """
    if checkpoint_path is None:
        checkpoint_path = CNN_BEST_MODEL
    if output_path is None:
        output_path = ONNX_MODEL_PATH

    ckpt_path = Path(checkpoint_path)
    onnx_path = Path(output_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPORT → ONNX")
    print("=" * 70)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output     : {onnx_path}")
    print(f"  Model type : {model_type}")

    # Load model
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = build_model(model_type, num_classes=NUM_ARRHYTHMIA_CLASSES,
                        num_channels=NUM_CHANNELS)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()

    # Dummy input: (1, 12, 2500) mV scale
    dummy = torch.randn(1, NUM_CHANNELS, WINDOW_SIZE) * 0.5

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=['ecg_input'],
        output_names=['arrhythmia_logits'],
        dynamic_axes={
            'ecg_input':          {0: 'batch'},
            'arrhythmia_logits':  {0: 'batch'},
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"\n✓ ONNX disimpan  : {onnx_path}")
    print(f"  Ukuran         : {onnx_path.stat().st_size/1024:.1f} KB")

    # Verifikasi dengan onnxruntime
    try:
        import onnxruntime as ort
        sess     = ort.InferenceSession(str(onnx_path),
                                        providers=['CPUExecutionProvider'])
        inp_name = sess.get_inputs()[0].name
        out      = sess.run(None, {inp_name: dummy.numpy()})[0]
        cls_pred = np.argmax(out, axis=1)[0]
        print(f"  Verifikasi ONNX: class={cls_pred}, flag={1<<cls_pred}")
        print("  ✓ Verifikasi berhasil")
    except ImportError:
        print("  ⚠ onnxruntime tidak tersedia, verifikasi dilewati.")

    return onnx_path


# ============================================================================
# EXPORT PKL
# ============================================================================

def export_to_pkl(checkpoint_path=None, output_path=None,
                  model_type: str = 'standard') -> Path:
    """
    Export model PyTorch → .pkl (HolterArrhythmiaModel wrapper).

    Returns:
        Path ke file .pkl
    """
    if checkpoint_path is None:
        checkpoint_path = CNN_BEST_MODEL
    if output_path is None:
        output_path = PKL_MODEL_PATH

    ckpt_path = Path(checkpoint_path)
    pkl_path  = Path(output_path)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPORT → .PKL")
    print("=" * 70)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output     : {pkl_path}")

    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = build_model(model_type, num_classes=NUM_ARRHYTHMIA_CLASSES,
                        num_channels=NUM_CHANNELS)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()

    wrapper = HolterArrhythmiaModel(model, device='cpu')

    with open(pkl_path, 'wb') as f:
        pickle.dump(wrapper, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✓ .pkl disimpan  : {pkl_path}")
    print(f"  Ukuran         : {pkl_path.stat().st_size/1024:.1f} KB")

    # Simpan metadata JSON
    meta = {
        'version':         'v3',
        'task':            'single-label classification (11 classes)',
        'num_classes':     NUM_ARRHYTHMIA_CLASSES,
        'num_channels':    NUM_CHANNELS,
        'window_size':     WINDOW_SIZE,
        'class_labels':    ARRHYTHMIA_LABELS,
        'class_to_flag':   {i: (1 << i) for i in range(NUM_ARRHYTHMIA_CLASSES)},
        'input_spec': {
            'shape':    f'(batch, {NUM_CHANNELS}, {WINDOW_SIZE})',
            'dtype':    'float32',
            'scale':    'mV  (NO normalization to [-1,1])',
            'note':     'Konsisten dengan ecg_signal_mv di ArrhythmiaDetector',
        },
        'output_spec': {
            'shape':    f'(batch, {NUM_ARRHYTHMIA_CLASSES})',
            'type':     'logits → argmax → class_index',
        },
        'arrhythmia_bin': {
            'format':   'one int32 per ECG sample',
            'value':    '1 << class_index  (= 2^class_index)',
            'dtype':    'np.int32',
            'normal':   'class 0 → flag=1=2^0 → dikecualikan parser (log2=0)',
            'aritmia':  'class 1–10 → flag 2–1024 → log2=1–10 → region aritmia',
        },
        'adc_gain_device': ADC_GAIN_DEVICE,
        'adc_gain_int16':  ADC_GAIN_INT16,
        'int16_to_mv':     INT16_TO_MV,
        'checkpoint':      str(ckpt_path),
        'model_type':      model_type,
    }
    meta_path = pkl_path.parent / "model_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata       : {meta_path}")

    return pkl_path


# ============================================================================
# INFERENCE: .bin → arrhythmia.bin
# ============================================================================

def run_inference(
    model_path,
    ecg_bin_path,
    output_dir=None,
    source:     str = 'device',     # 'device'|'int16'|'mv'
    batch_size: int = 32,
):
    """
    Jalankan inference pada satu rekaman ECG .bin → arrhythmia.bin.

    Args:
        model_path   : Path ke .pkl atau .onnx model.
        ecg_bin_path : Path ke ECG .bin (format Holter int16).
        output_dir   : Direktori output (default: parent ecg_bin_path).
        source       : Format data input:
                         'device' → raw ADC int16, ×0.0025 → mV
                         'int16'  → int16, /1000   → mV
                         'mv'     → sudah float32 mV (jarang)
        batch_size   : Batch size inference.

    Output:
        arrhythmia.bin: one int32 per ECG sample = (1 << class_index)
        Kompatibel langsung dengan arrhythmia_parser.py di app.
    """
    ecg_path   = Path(ecg_bin_path)
    output_dir = Path(output_dir) if output_dir else ecg_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INFERENCE  ECG .bin → arrhythmia.bin")
    print("=" * 70)
    print(f"  Model  : {model_path}")
    print(f"  Input  : {ecg_path.name}")
    print(f"  Source : {source}")

    # Baca ECG
    raw  = np.fromfile(ecg_path, dtype=np.int16)
    trim = len(raw) - (len(raw) % NUM_CHANNELS)
    raw  = raw[:trim]
    ecg  = raw.reshape(-1, NUM_CHANNELS)       # (n_samples, 12) int16

    # Konversi ke mV sesuai source
    if source == 'device':
        ecg_mv = ecg.astype(np.float32) * RAW_ADC_TO_MV
    elif source == 'int16':
        ecg_mv = ecg.astype(np.float32) * INT16_TO_MV
    elif source == 'mv':
        ecg_mv = ecg.astype(np.float32)
    else:
        raise ValueError(f"source harus 'device', 'int16', atau 'mv'")

    n_samples = ecg_mv.shape[0]
    duration  = n_samples / HOLTER_SAMPLING_RATE
    print(f"  Samples: {n_samples:,}  ({duration:.1f}s = {duration/60:.1f} menit)")

    # Load model
    model_p = Path(model_path)
    if model_p.suffix == '.pkl':
        with open(model_p, 'rb') as f:
            wrapper: HolterArrhythmiaModel = pickle.load(f)
        flags = wrapper.infer_recording(ecg_mv, batch_size=batch_size)

    elif model_p.suffix == '.onnx':
        import onnxruntime as ort
        sess     = ort.InferenceSession(str(model_p),
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        inp_name = sess.get_inputs()[0].name
        flags    = _onnx_infer_recording(sess, inp_name, ecg_mv, batch_size)

    else:
        raise ValueError(f"Model format tidak dikenal: {model_p.suffix}")

    # Verifikasi panjang
    assert len(flags) == n_samples, \
        f"Panjang flags {len(flags)} ≠ n_samples {n_samples}"

    # Tulis arrhythmia.bin
    arrhythmia_bin = output_dir / "arrhythmia.bin"
    flags.tofile(arrhythmia_bin)

    # Ringkasan deteksi
    print(f"\n  arrhythmia.bin  : {arrhythmia_bin}")
    print(f"  Format          : int32 per sample, n={n_samples:,}, "
          f"ukuran={arrhythmia_bin.stat().st_size/1024:.1f} KB")

    cls_idx, sample_counts = np.unique(flags, return_counts=True)
    print("\n  Distribusi kelas:")
    for f_val, cnt in zip(cls_idx, sample_counts):
        if f_val <= 0:
            continue
        c    = int(np.log2(f_val)) if f_val > 0 else 0
        name = ARRHYTHMIA_CLASSES.get(c, f"?({c})")
        sec  = cnt / HOLTER_SAMPLING_RATE
        print(f"    Class {c:2d} [{name:22s}] flag={f_val:<5d}: "
              f"{cnt:7,} samples = {sec:.1f}s")

    return flags


def _onnx_infer_recording(sess, inp_name: str,
                            ecg_mv: np.ndarray,
                            batch_size: int) -> np.ndarray:
    """Helper: inference rekaman lengkap via ONNX session."""
    import onnxruntime as ort  # noqa (sudah tersedia jika sampai sini)

    n_samples = ecg_mv.shape[0]
    remainder = n_samples % WINDOW_SIZE
    pad_len   = (WINDOW_SIZE - remainder) % WINDOW_SIZE
    padded    = (np.pad(ecg_mv, ((0, pad_len), (0, 0)))
                 if pad_len else ecg_mv)

    total_win = padded.shape[0] // WINDOW_SIZE
    all_cls   = []

    for i in range(0, total_win, batch_size):
        end   = min(i + batch_size, total_win)
        bsz   = end - i
        chunk = padded[i*WINDOW_SIZE : end*WINDOW_SIZE]
        inp   = (chunk.reshape(bsz, WINDOW_SIZE, NUM_CHANNELS)
                      .transpose(0, 2, 1)
                      .astype(np.float32))
        out   = sess.run(None, {inp_name: inp})[0]
        all_cls.append(np.argmax(out, axis=1).astype(np.int32))

    window_cls    = np.concatenate(all_cls)
    sample_cls    = np.repeat(window_cls, WINDOW_SIZE)
    if pad_len:
        sample_cls = sample_cls[:-pad_len]

    return (1 << sample_cls).astype(np.int32)


# ============================================================================
# BATCH INFERENCE
# ============================================================================

def run_inference_batch(model_path, ecg_dir, output_dir=None,
                         source='device', batch_size=32):
    """
    Inference seluruh file .bin dalam satu folder ECG rekaman.
    Menghasilkan satu arrhythmia.bin per sub-folder rekaman.
    """
    ecg_dir    = Path(ecg_dir)
    output_dir = Path(output_dir) if output_dir else ecg_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(ecg_dir.rglob("*.bin"))
    if not bin_files:
        print(f"⚠ Tidak ada .bin di {ecg_dir}")
        return {}

    print("=" * 70)
    print(f"BATCH INFERENCE: {len(bin_files)} file")
    print("=" * 70)

    results = {}
    for ecg_file in tqdm(bin_files, desc="Batch inference"):
        try:
            flags = run_inference(
                model_path=model_path,
                ecg_bin_path=ecg_file,
                output_dir=output_dir / ecg_file.stem,
                source=source,
                batch_size=batch_size,
            )
            unique_cls = set(int(np.log2(f)) for f in np.unique(flags) if f > 1)
            results[ecg_file.name] = {
                'success': True,
                'n_samples': len(flags),
                'detected_classes': sorted(unique_cls),
            }
        except Exception as e:
            print(f"  ✗ Error {ecg_file.name}: {e}")
            results[ecg_file.name] = {'success': False, 'error': str(e)}

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Ringkasan disimpan: {summary_path}")
    return results


# ============================================================================
# PIPELINE EXPORT LENGKAP
# ============================================================================

def export_pipeline(checkpoint_path=None, output_dir=None,
                    model_type='standard'):
    """
    Export ONNX + PKL sekaligus.
    """
    if output_dir is None:
        output_dir = EXPORTED_MODELS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output_dir / "arrhythmia_model.onnx",
        model_type=model_type,
    )
    pkl_path = export_to_pkl(
        checkpoint_path=checkpoint_path,
        output_path=output_dir / "arrhythmia_model.pkl",
        model_type=model_type,
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print(f"  ONNX : {onnx_path}  ← dipakai langsung oleh ArrhythmiaDetector")
    print(f"  PKL  : {pkl_path}   ← untuk testing / integrasi custom")
    print("=" * 70)

    return onnx_path, pkl_path


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export model & inference ECG Holter → arrhythmia.bin"
    )
    sub = parser.add_subparsers(dest='cmd')

    # export
    pe = sub.add_parser('export', help='Export ONNX + PKL')
    pe.add_argument('--checkpoint',  default=None)
    pe.add_argument('--output-dir',  default=None)
    pe.add_argument('--model-type',  choices=['standard','improved'],
                    default='standard')

    # infer
    pi = sub.add_parser('infer', help='Inference satu rekaman')
    pi.add_argument('--model',  required=True, help='.onnx atau .pkl')
    pi.add_argument('--input',  required=True, help='ECG .bin')
    pi.add_argument('--output', default=None)
    pi.add_argument('--source', choices=['device','int16','mv'], default='device')
    pi.add_argument('--batch-size', type=int, default=32)

    # batch
    pb = sub.add_parser('batch', help='Inference folder rekaman')
    pb.add_argument('--model',  required=True)
    pb.add_argument('--input',  required=True, help='Folder berisi .bin')
    pb.add_argument('--output', default=None)
    pb.add_argument('--source', choices=['device','int16','mv'], default='device')
    pb.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()

    if args.cmd == 'export':
        export_pipeline(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            model_type=args.model_type,
        )
    elif args.cmd == 'infer':
        run_inference(
            model_path=args.model,
            ecg_bin_path=args.input,
            output_dir=args.output,
            source=args.source,
            batch_size=args.batch_size,
        )
    elif args.cmd == 'batch':
        run_inference_batch(
            model_path=args.model,
            ecg_dir=args.input,
            output_dir=args.output,
            source=args.source,
            batch_size=args.batch_size,
        )
    else:
        parser.print_help()
