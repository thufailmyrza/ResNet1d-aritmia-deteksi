"""
model/resnet1d.py  –  BARU v3
ResNet-1D dengan Squeeze-Excitation Attention untuk deteksi aritmia 11 kelas.

Arsitektur:
  Input: (batch, 12, 2500)  –  12-lead ECG, 5 detik @ 500 Hz, skala mV
  Stem   → conv 1D, stride 2  → (B, 64, 1250)
  Block1 → 2× ResBlock(64→64)  → (B, 64, 1250)
  Block2 → 2× ResBlock(64→128, stride=2) → (B, 128, 625)
  Block3 → 2× ResBlock(128→256, stride=2) → (B, 256, 313)
  Block4 → 2× ResBlock(256→512, stride=2) → (B, 512, 157)
  GAP    → (B, 512)
  Head   → Linear 512→256 → BN → ReLU → Dropout → Linear 256→11

Setiap ResBlock mengandung SE (Squeeze-Excitation) attention per-channel.
Output: logits (B, 11) – gunakan argmax untuk prediksi kelas.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
#  Path setup 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
# BUILDING BLOCKS
class SEBlock(nn.Module):
    """
    Squeeze-Excitation Block (channel attention).
    Memperkuat fitur channel yang relevan dan menekan yang tidak.

    Paper: "Squeeze-and-Excitation Networks" (Hu et al., 2018).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze   = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        s = self.squeeze(x).squeeze(-1)          # (B, C)
        e = self.excitation(s).unsqueeze(-1)     # (B, C, 1)
        return x * e                              # channel-wise scaling


class ResBlock1D(nn.Module):
    """
    Basic Residual Block untuk 1-D ECG signal.
    Struktur: Conv → BN → ReLU → Conv → BN → SE → Add shortcut → ReLU
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 se_reduction: int = 16, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7,
                               padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.se      = SEBlock(out_ch, reduction=se_reduction)
        self.dropout = nn.Dropout(dropout)

        # Shortcut (projection jika dimensi berubah)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out = out + identity
        return F.relu(out, inplace=True)


class BottleneckBlock1D(nn.Module):
    """
    Bottleneck Residual Block (untuk jaringan lebih dalam).
    1×1 → 3×3 → 1×1, lebih efisien untuk jumlah parameter.
    Dipakai opsional jika bottleneck=True di ResNet1D.
    """
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1,
                 se_reduction: int = 16, dropout: float = 0.1):
        super().__init__()
        out_ch = mid_ch * self.expansion

        self.conv1 = nn.Conv1d(in_ch, mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(mid_ch)

        self.conv2 = nn.Conv1d(mid_ch, mid_ch, 7, stride=stride,
                               padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(mid_ch)

        self.conv3 = nn.Conv1d(mid_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm1d(out_ch)

        self.se      = SEBlock(out_ch, reduction=se_reduction)
        self.dropout = nn.Dropout(dropout)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        return F.relu(out + identity, inplace=True)

# MULTI-SCALE TEMPORAL ATTENTION
class MultiScaleAttention(nn.Module):
    """
    Multi-scale temporal attention: rata-rata + max pooling di berbagai
    kernel size, diikuti 1D conv untuk menghasilkan attention mask.
    Membantu model fokus pada segmen waktu yang paling informatif.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)   # (B, 1, L)
        mx  = x.max(dim=1, keepdim=True).values
        combined = torch.cat([avg, mx], dim=1)   # (B, 2, L)
        attn = self.conv(combined)               # (B, 1, L)
        return x * attn

# RESNET-1D MODEL
class ResNet1D(nn.Module):
    """
    ResNet-1D untuk klasifikasi aritmia ECG 12-lead, single-label.

    Args:
        num_classes   : Jumlah kelas output (default 11).
        num_channels  : Jumlah lead ECG (default 12).
        base_filters  : Jumlah filter di stem & block pertama (default 64).
        layers        : Jumlah ResBlock per stage [2, 2, 2, 2].
        dropout       : Dropout rate (default 0.3).
        se_reduction  : SE reduction ratio (default 16).
        use_attention : Aktifkan multi-scale temporal attention (default True).
    """

    def __init__(
        self,
        num_classes:   int   = None,
        num_channels:  int   = 12,
        base_filters:  int   = 64,
        layers:        list  = None,
        dropout:       float = 0.3,
        se_reduction:  int   = 16,
        use_attention: bool  = True,
    ):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]

        self.num_classes = num_classes

        # ── Stem ────────────────────────────────────────────────────────────
        # Input: (B, 12, 2500) → (B, 64, 1250)
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, base_filters, kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )

        # ── Residual Stages ──────────────────────────────────────────────────
        f0, f1, f2, f3 = (base_filters,
                          base_filters * 2,
                          base_filters * 4,
                          base_filters * 8)

        self.stage1 = self._make_stage(f0, f0, layers[0], stride=1,
                                       se_reduction=se_reduction,
                                       dropout=dropout)
        self.stage2 = self._make_stage(f0, f1, layers[1], stride=2,
                                       se_reduction=se_reduction,
                                       dropout=dropout)
        self.stage3 = self._make_stage(f1, f2, layers[2], stride=2,
                                       se_reduction=se_reduction,
                                       dropout=dropout)
        self.stage4 = self._make_stage(f2, f3, layers[3], stride=2,
                                       se_reduction=se_reduction,
                                       dropout=dropout)

        # ── Temporal Attention (opsional) ────────────────────────────────────
        self.use_attention = use_attention
        if use_attention:
            self.temporal_attn = MultiScaleAttention(f3)

        # ── Classification Head ──────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(f3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # ── Weight Init ──────────────────────────────────────────────────────
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int,
                    stride: int, se_reduction: int,
                    dropout: float) -> nn.Sequential:
        blocks = [ResBlock1D(in_ch, out_ch, stride=stride,
                             se_reduction=se_reduction, dropout=dropout)]
        for _ in range(1, n_blocks):
            blocks.append(ResBlock1D(out_ch, out_ch, stride=1,
                                     se_reduction=se_reduction,
                                     dropout=dropout))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, 2500) float32  –  ECG dalam mV

        Returns:
            logits: (B, 11) float32  –  raw logits (BELUM softmax)
        """
        x = self.stem(x)        # (B, 64, 1250)
        x = self.stage1(x)      # (B, 64, 1250)
        x = self.stage2(x)      # (B, 128, 625)
        x = self.stage3(x)      # (B, 256, 313)
        x = self.stage4(x)      # (B, 512, 157)

        if self.use_attention:
            x = self.temporal_attn(x)

        x = self.gap(x).squeeze(-1)   # (B, 512)
        return self.head(x)            # (B, 11)

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return class index (argmax). Shape: (B,) int64.
        Selalu berjalan dalam eval mode (dropout dinonaktifkan) meski
        model sedang dalam training mode – state dikembalikan setelahnya.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            result = logits.argmax(dim=1)
        if was_training:
            self.train()
        return result

    def predict_flag(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return flag values: (1 << class_index) per sample.
        Cocok langsung dengan arrhythmia.bin format.
        Shape: (B,) int32.

        Hanya satu forward pass (lewat predict_class) – tidak ada
        risiko hasil berbeda akibat dropout.
        """
        cls = self.predict_class(x).cpu()   # eval mode, no_grad
        return (1 << cls).to(torch.int32)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# IMPROVED (DEEPER) VARIANT
class ImprovedResNet(ResNet1D):
    """
    Varian ResNet-1D yang lebih dalam: [3, 4, 6, 3] blocks.
    Cocok untuk dataset lebih besar.
    """

    def __init__(self, num_classes: int = 11, num_channels: int = 12,
                 dropout: float = 0.3):
        super().__init__(
            num_classes=num_classes,
            num_channels=num_channels,
            base_filters=64,
            layers=[3, 4, 6, 3],
            dropout=dropout,
            se_reduction=16,
            use_attention=True,
        )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================
def build_model(model_type: str = 'standard',
                num_classes: int = 11,
                num_channels: int = 12,
                dropout: float = 0.3):
    """
    Factory function untuk membuat model berdasarkan tipe.

    Args:
        model_type   : 'standard'  → ResNet-18 style (SE-ResBlock [2,2,2,2])
                       'improved'  → ResNet-50 style (SE-ResBlock [3,4,6,3])
                       'resnet152' → Bottleneck ResNet dari notebook (file ini)
        num_classes  : Jumlah kelas output (default 11)
        num_channels : Jumlah lead ECG (default 12)
        dropout      : Dropout rate

    Returns:
        Model instance (belum dilatih, belum .to(device))
    """
    if model_type == 'improved':
        model = ImprovedResNet(num_classes=num_classes,
                               num_channels=num_channels,
                               dropout=dropout)
        n_params = model.count_parameters()
        print(f"  Model: {model_type}  |  Params: {n_params:,}")

    elif model_type == 'resnet152':
        from model.resnet152 import build_resnet152
        model = build_resnet152(
            num_classes=num_classes,
            num_channels=num_channels,
            dropout=dropout,
        )

    else:  # 'standard'
        model = ResNet1D(num_classes=num_classes,
                         num_channels=num_channels,
                         dropout=dropout)
        n_params = model.count_parameters()
        print(f"  Model: {model_type}  |  Params: {n_params:,}")

    return model

# SELF-TEST
if __name__ == "__main__":
    print("=" * 60)
    print("SELF-TEST: ResNet1D")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    for mtype in ['standard', 'improved', 'resnet152']:
        print(f"\n── {mtype} ──")
        model = build_model(mtype).to(device)

        # Dummy input: batch=4, 12 leads, 2500 samples (mV scale)
        x = torch.randn(4, 12, 2500, device=device) * 0.5   # ~0.5 mV std

        # ── WAJIB: set eval sebelum prediksi ─────────────────────────────
        # Tanpa ini, Dropout aktif dan setiap forward() hasilkan output
        # berbeda → predict_class dan predict_flag tidak konsisten.
        model.eval()

        with torch.no_grad():
            # Satu forward pass → logits, preds, flags semuanya konsisten
            logits = model(x)
            preds  = logits.argmax(dim=1)                       # (4,) int64
            flags  = (1 << preds.cpu()).to(torch.int32)         # (4,) int32

        print(f"  Input  : {tuple(x.shape)}")
        print(f"  Logits : {tuple(logits.shape)}  dtype={logits.dtype}")
        print(f"  Preds  : {preds.tolist()}  (class index 0–10)")
        print(f"  Flags  : {flags.tolist()}   (= 1 << class_idx)")

        # Verifikasi flag = 2^class_idx
        for c, f in zip(preds.tolist(), flags.tolist()):
            expected = 1 << c
            assert f == expected, \
                f"Flag mismatch: got {f}, expected 1<<{c}={expected}"
        print("  ✓ Flag = 2^class_idx verified")

        # Verifikasi predict_class & predict_flag konsisten dengan single-pass
        model.train()   # kembalikan ke training mode untuk tes konsistensi
        pc    = model.predict_class(x)            # gunakan eval() internal
        pf    = model.predict_flag(x)             # gunakan predict_class internal
        flags2 = (1 << pc.cpu()).to(torch.int32)

        assert torch.equal(pc.cpu(), preds.cpu()), \
            f"predict_class tidak konsisten dengan logits.argmax:\n  {pc.tolist()} vs {preds.tolist()}"
        assert torch.equal(pf.cpu(), flags2.cpu()), \
            f"predict_flag tidak konsisten dengan predict_class:\n  {pf.tolist()} vs {flags2.tolist()}"
        print("  ✓ predict_class & predict_flag konsisten (eval mode guard ok)")

        # Pastikan model kembali ke training mode setelah predict_*
        assert model.training, "Model tidak kembali ke training mode setelah predict_class!"
        print("  ✓ Training mode dipulihkan setelah predict_class")

    print("\n" + "=" * 60)
    print("✓ Self-test passed")
    print("=" * 60)