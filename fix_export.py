# fix_export_single_file.py
# Jalankan di: Project Arrythmia/ (bukan folder aplikasi)

import torch
import onnx
from onnxmltools.utils import save_model
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from config_path import (
    CNN_BEST_MODEL, EXPORTED_MODELS_DIR,
    NUM_ARRHYTHMIA_CLASSES, NUM_CHANNELS, WINDOW_SIZE
)
from model.resnet1d import build_model

# ── Load checkpoint ──────────────────────────────────────────────────
CHECKPOINT  = CNN_BEST_MODEL   # atau ganti path manual
MODEL_TYPE  = "resnet152"      # sesuaikan: standard / improved / resnet152
OUTPUT_PATH = EXPORTED_MODELS_DIR / "arrhythmia_model_single.onnx"

print(f"Loading checkpoint: {CHECKPOINT}")
ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = build_model(MODEL_TYPE,
                    num_classes=NUM_ARRHYTHMIA_CLASSES,
                    num_channels=NUM_CHANNELS)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()

# ── Export TANPA external data ───────────────────────────────────────
dummy = torch.randn(1, NUM_CHANNELS, WINDOW_SIZE) * 0.5

print("Exporting ke ONNX single-file...")
torch.onnx.export(
    model, dummy, str(OUTPUT_PATH),
    input_names=["ecg_input"],
    output_names=["arrhythmia_logits"],
    dynamic_axes={
        "ecg_input":         {0: "batch"},
        "arrhythmia_logits": {0: "batch"},
    },
    opset_version=14,
    do_constant_folding=True,
    export_params=True,
    # Kunci utama: TIDAK ada parameter external_data
)

# ── Verifikasi tidak ada .data file ─────────────────────────────────
data_file = Path(str(OUTPUT_PATH) + ".data")
if data_file.exists():
    # Kalau masih terbuat, paksa inline dengan onnx library
    print("Model masih punya .data, menggabungkan ke satu file...")
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
    
    model_proto = onnx.load(str(OUTPUT_PATH))
    load_external_data_for_model(model_proto, str(OUTPUT_PATH.parent))
    
    # Simpan ulang sebagai inline (semua bobot di dalam .onnx)
    onnx.save(model_proto, str(OUTPUT_PATH),
              save_as_external_data=False)
    
    # Hapus .data yang tidak diperlukan lagi
    data_file.unlink()
    print(f"File .data dihapus: {data_file}")

size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
print(f"\nHasil: {OUTPUT_PATH}")
print(f"Ukuran: {size_mb:.1f} MB  (single file, tidak ada .data)")

# ── Quick verify ─────────────────────────────────────────────────────
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(str(OUTPUT_PATH),
                             providers=["CPUExecutionProvider"])
x    = np.random.randn(1, NUM_CHANNELS, WINDOW_SIZE).astype(np.float32) * 0.3
out  = sess.run(None, {"ecg_input": x})[0]
cls  = int(out.argmax())

print(f"\nVerifikasi inference:")
print(f"  Output shape : {out.shape}")   # (1, 11)
print(f"  Predicted    : class {cls}  →  flag {1 << cls}")
print(f"\n✓ SIAP disalin ke resource/models/detector.onnx")