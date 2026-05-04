# verify_my_onnx.py
import onnxruntime as ort
import numpy as np

from config_path import ONNX_MODEL_PATH

sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

print("=== INPUT NODES ===")
for inp in sess.get_inputs():
    print(f"  name : '{inp.name}'")
    print(f"  shape: {inp.shape}")
    print(f"  dtype: {inp.type}")

print("\n=== OUTPUT NODES ===")
for out in sess.get_outputs():
    print(f"  name : '{out.name}'")
    print(f"  shape: {out.shape}")

# Dummy inference
dummy = np.random.randn(1, 12, 2500).astype(np.float32) * 0.5
result = sess.run(None, {sess.get_inputs()[0].name: dummy})
print(f"\n=== OUTPUT SHAPE: {result[0].shape} ===")   # harus (1, 11)
print(f"Predicted class: {np.argmax(result[0], axis=1)}")

# Sanity: loss awal random model ≈ ln(11) = 2.398
import torch, torch.nn.functional as F
logits = torch.tensor(result[0])
loss = F.cross_entropy(logits, torch.tensor([0]))
print(f"Dummy CE loss: {loss.item():.3f}  (random init → expected ≈ 2.40)")