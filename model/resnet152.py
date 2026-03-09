"""
model/model_resnet152.py
========================
Arsitektur ResNet1D Bottleneck (dari notebook ResNet152.ipynb),
dikonversi ke PyTorch dan diintegrasikan penuh dengan pipeline project:

  • Interface identik dengan ResNet1D di resnet1d.py
    (predict_class, predict_flag, count_parameters)
  • Diregistrasi di build_model() via model_type='resnet152'
  • Dipakai di train_model.py dengan flag --model-type resnet152

Input  : (batch, 12, 2500)  — 12-lead, 500 Hz, 5 detik, skala mV
Output : (batch, 11) logits — 11 kelas aritmia

Perbandingan arsitektur:
┌──────────────┬────────────────────────────────┬────────────────────────────┐
│              │ Model LAMA (resnet1d.py)        │ Model BARU (file ini)      │
├──────────────┼────────────────────────────────┼────────────────────────────┤
│ Block type   │ SE-ResBlock (2-layer wide)      │ Bottleneck (3-layer deep)  │
│ Attention    │ Temporal multi-scale attention  │ -                          │
│ Kernel stem  │ k=15                            │ k=7                        │
│ Stage layout │ [2,2,2,2] atau [3,4,6,3]        │ [3,4,6,3] fixed            │
│ Head         │ Linear 512→256→11              │ Linear 2048→256→128→11     │
│ Init         │ default PyTorch                 │ Xavier Uniform (Glorot)    │
└──────────────┴────────────────────────────────┴────────────────────────────┘
"""

from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config_path import NUM_ARRHYTHMIA_CLASSES


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class IdentityBlock1D(nn.Module):
    """
    Bottleneck Identity Block — shortcut tanpa perubahan dimensi channel/length.
    Setara identity_block() di notebook Keras.

    Path utama : Conv(F1,k=1) → BN → ReLU
                 Conv(F2,k=f) → BN → ReLU
                 Conv(F3,k=1) → BN
    Shortcut   : langsung (dimensi tidak berubah)
    Output     : Add → ReLU
    """

    def __init__(self, in_channels: int, filters: list[int], f: int = 3):
        """
        Args:
            in_channels : channel masuk (= F3 dari block sebelumnya)
            filters     : [F1, F2, F3]
            f           : kernel_size untuk conv tengah
        """
        super().__init__()
        F1, F2, F3 = filters
        assert in_channels == F3, (
            f"IdentityBlock: in_channels ({in_channels}) harus == F3 ({F3})"
        )

        self.main = nn.Sequential(
            # 1×1 bottleneck masuk
            nn.Conv1d(in_channels, F1, kernel_size=1, bias=False),
            nn.BatchNorm1d(F1),
            nn.ReLU(inplace=True),
            # f×1 conv utama
            nn.Conv1d(F1, F2, kernel_size=f, padding=f // 2, bias=False),
            nn.BatchNorm1d(F2),
            nn.ReLU(inplace=True),
            # 1×1 bottleneck keluar
            nn.Conv1d(F2, F3, kernel_size=1, bias=False),
            nn.BatchNorm1d(F3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(x) + x, inplace=True)


class ConvolutionalBlock1D(nn.Module):
    """
    Bottleneck Convolutional Block — shortcut di-projection untuk menyamakan dimensi.
    Setara convolutional_block() di notebook Keras.

    Path utama : Conv(F1,k=1,s=s) → BN → ReLU
                 Conv(F2,k=f,s=1) → BN → ReLU
                 Conv(F3,k=1,s=1) → BN
    Shortcut   : Conv(F3,k=1,s=s) → BN
    Output     : Add → ReLU
    """

    def __init__(self, in_channels: int, filters: list[int],
                 f: int = 3, s: int = 2):
        """
        Args:
            in_channels : channel masuk
            filters     : [F1, F2, F3]
            f           : kernel_size untuk conv tengah
            s           : stride untuk conv pertama dan shortcut
        """
        super().__init__()
        F1, F2, F3 = filters

        self.main = nn.Sequential(
            nn.Conv1d(in_channels, F1, kernel_size=1, stride=s, bias=False),
            nn.BatchNorm1d(F1),
            nn.ReLU(inplace=True),

            nn.Conv1d(F1, F2, kernel_size=f, padding=f // 2, bias=False),
            nn.BatchNorm1d(F2),
            nn.ReLU(inplace=True),

            nn.Conv1d(F2, F3, kernel_size=1, bias=False),
            nn.BatchNorm1d(F3),
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, F3, kernel_size=1, stride=s, bias=False),
            nn.BatchNorm1d(F3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(x) + self.shortcut(x), inplace=True)


# ============================================================================
# BACKBONE
# ============================================================================

class ResNet152Backbone(nn.Module):
    """
    Backbone ResNet Bottleneck 1D.
    Stage layout identik dengan notebook, disesuaikan untuk ECG 12-lead.

    Stem  : ZeroPad(3) → Conv(64,k=7,s=2) → BN → ReLU → MaxPool(3,s=2)
            Output: (B, 64, ~624)

    Stage2: ConvBlock([128,128,256], s=1) + 2× IdentityBlock
            Output: (B, 256, ~624)

    Stage3: ConvBlock([128,128,512], s=2) + 3× IdentityBlock
            Output: (B, 512, ~312)

    Stage4: ConvBlock([256,256,1024],s=2) + 5× IdentityBlock
            Output: (B,1024, ~156)

    Stage5: ConvBlock([512,512,2048],s=2) + 2× IdentityBlock
            Output: (B,2048, ~78)

    Pool  : AdaptiveAvgPool1d(1) → squeeze
            Output: (B, 2048)
    """

    def __init__(self, in_channels: int = 12):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.ZeroPad1d(3),
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # ── Stage 2  (in=64 → out=256) ────────────────────────────────────
        self.stage2 = nn.Sequential(
            ConvolutionalBlock1D(64,  [128, 128, 256], f=3, s=1),
            IdentityBlock1D(256, [128, 128, 256], f=3),
            IdentityBlock1D(256, [128, 128, 256], f=3),
        )

        # ── Stage 3  (in=256 → out=512) ───────────────────────────────────
        self.stage3 = nn.Sequential(
            ConvolutionalBlock1D(256, [128, 128, 512], f=3, s=2),
            IdentityBlock1D(512, [128, 128, 512], f=3),
            IdentityBlock1D(512, [128, 128, 512], f=3),
            IdentityBlock1D(512, [128, 128, 512], f=3),
        )

        # ── Stage 4  (in=512 → out=1024) ──────────────────────────────────
        self.stage4 = nn.Sequential(
            ConvolutionalBlock1D(512,  [256, 256, 1024], f=3, s=2),
            IdentityBlock1D(1024, [256, 256, 1024], f=3),
            IdentityBlock1D(1024, [256, 256, 1024], f=3),
            IdentityBlock1D(1024, [256, 256, 1024], f=3),
            IdentityBlock1D(1024, [256, 256, 1024], f=3),
            IdentityBlock1D(1024, [256, 256, 1024], f=3),
        )

        # ── Stage 5  (in=1024 → out=2048) ─────────────────────────────────
        self.stage5 = nn.Sequential(
            ConvolutionalBlock1D(1024, [512, 512, 2048], f=3, s=2),
            IdentityBlock1D(2048, [512, 512, 2048], f=3),
            IdentityBlock1D(2048, [512, 512, 2048], f=3),
        )

        # ── Global Average Pool ────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)

        self._init_weights()

    def _init_weights(self):
        """Xavier Uniform (Glorot) — sama dengan notebook Keras."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.gap(x).squeeze(-1)   # (B, 2048)
        return x


# ============================================================================
# FULL MODEL
# ============================================================================

class ResNet152ECG(nn.Module):
    """
    ResNet Bottleneck 1D untuk deteksi aritmia 11 kelas.

    Interface identik dengan ResNet1D di resnet1d.py:
      • forward(x)         → logits (B, num_classes)
      • predict_class(x)   → class_index (B,) int64
      • predict_flag(x)    → 1 << class_index (B,) int32  [format arrhythmia.bin]
      • count_parameters() → int

    Args:
        num_classes  : jumlah kelas (default 11 dari config_path)
        num_channels : jumlah lead ECG (default 12)
        dropout_rate : dropout sebelum head (default 0.5)
    """

    def __init__(
        self,
        num_classes:  int   = NUM_ARRHYTHMIA_CLASSES,
        num_channels: int   = 12,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.num_channels = num_channels

        self.backbone = ResNet152Backbone(in_channels=num_channels)

        # Head identik dengan notebook: FC256 → FC128 → FC(num_classes)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, num_channels, seq_len)  mis. (B, 12, 2500)
        Returns:
            logits : (B, num_classes)   — belum softmax, pakai CrossEntropyLoss
        """
        return self.head(self.backbone(x))

    # ── Interface yang dibutuhkan train_model.py & inference_export_pkl.py ──

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediksi class index. Selalu berjalan dalam eval mode.

        Returns:
            (B,) int64
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            result = self.forward(x).argmax(dim=1)
        if was_training:
            self.train()
        return result

    def predict_flag(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediksi dalam format flag bit arrhythmia.bin: (1 << class_index).

        Returns:
            (B,) int32
        """
        return (1 << self.predict_class(x).cpu()).to(torch.int32)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# FACTORY — dipanggil oleh build_model() di resnet1d.py
# ============================================================================

def build_resnet152(
    num_classes:  int   = NUM_ARRHYTHMIA_CLASSES,
    num_channels: int   = 12,
    dropout:      float = 0.5,
) -> ResNet152ECG:
    """
    Factory function dipanggil dari build_model() di resnet1d.py
    ketika model_type == 'resnet152'.

    Contoh pemakaian langsung:
        from model.model_resnet152 import build_resnet152
        model = build_resnet152()
    """
    model = ResNet152ECG(
        num_classes=num_classes,
        num_channels=num_channels,
        dropout_rate=dropout,
    )
    print(f"  Model: resnet152  |  Params: {model.count_parameters():,}")
    return model


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import math

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("SELF-TEST: ResNet152ECG")
    print("=" * 60)
    print(f"Device : {device}\n")

    model = build_resnet152().to(device)

    # Dummy input: batch=4, 12 lead, 2500 sampel (mV scale ~0.5 mV std)
    x = torch.randn(4, 12, 2500, device=device) * 0.5

    # ── forward ──
    model.eval()
    with torch.no_grad():
        logits = model(x)
        cls    = model.predict_class(x)
        flags  = model.predict_flag(x)
        loss   = nn.CrossEntropyLoss()(logits, cls)

    print(f"Input shape    : {tuple(x.shape)}")
    print(f"Output (logits): {tuple(logits.shape)}")   # harus (4, 11)
    print(f"Classes        : {cls.tolist()}")
    print(f"Flags          : {flags.tolist()}")
    print(f"Initial loss   : {loss.item():.4f}  (expected ≈ {math.log(11):.4f})")

    assert logits.shape == (4, 11), "Output shape salah!"
    assert all(0 <= c < 11 for c in cls.tolist()), "Class index di luar range!"
    print("\n✓ Semua assertion passed.")

    # ── Parameter count ──
    total = model.count_parameters()
    print(f"\nTotal params     : {total:,}")
    print(f"Model size (fp32): ~{total * 4 / 1024**2:.1f} MB")