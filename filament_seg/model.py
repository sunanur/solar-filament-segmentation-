"""Segmentation architectures and multi-head loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

__all__ = [
    "EdgeGuidedAttention",
    "EdgeAttResUNet3Head",
    "ConvNeXtAttUNet3Head",
    "multi_head_loss",
    "load_model",
]


class EdgeGuidedAttention(nn.Module):
    """Multi-head self-attention gated by a learned spatial edge map."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.edge_q = nn.Linear(1, channels)
        self.edge_k = nn.Linear(1, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, edge_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        xf = x.view(b, c, n).permute(0, 2, 1)
        ef = edge_map.view(b, 1, n).permute(0, 2, 1)

        q = (self.q_proj(xf) * self.edge_q(ef)).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = (self.k_proj(xf) * self.edge_k(ef)).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(xf).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, n, c)
        out = self.norm(self.out_proj(attn) + xf)
        return out.permute(0, 2, 1).view(b, c, h, w)


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeAttResUNet3Head(nn.Module):
    """ResNet-34 U-Net with bottleneck edge-guided attention and 3 output heads (mask, hull, spine)."""

    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        enc = models.resnet34(weights=weights)

        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.layer1 = enc.layer1
        self.layer2 = enc.layer2
        self.layer3 = enc.layer3
        self.layer4 = enc.layer4

        self.eg_mhsa = EdgeGuidedAttention(channels=512, num_heads=8)

        self.dec4 = ConvBlock(512 + 256, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 64, 64)
        self.dec0 = ConvBlock(64 + in_channels, 48)

        self.head_mask = nn.Conv2d(48, 1, kernel_size=1)
        self.head_hull = nn.Conv2d(48, 1, kernel_size=1)
        self.head_spine = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = self.edge_branch(x)
        s0 = self.stem(x)
        e1 = self.layer1(self.pool(s0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        edge_low = F.interpolate(edge, size=e4.shape[2:], mode="bilinear", align_corners=False)
        b = self.eg_mhsa(e4, edge_low)

        d4 = self.dec4(torch.cat([F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False), e3], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=e2.shape[2:], mode="bilinear", align_corners=False), e2], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e1.shape[2:], mode="bilinear", align_corners=False), e1], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=s0.shape[2:], mode="bilinear", align_corners=False), s0], dim=1))
        d0 = self.dec0(torch.cat([F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False), x], dim=1))

        return torch.cat([self.head_mask(d0), self.head_hull(d0), self.head_spine(d0)], dim=1)


class ASPPBridge(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid = out_c // 2
        self.b1 = nn.Sequential(nn.Conv2d(in_c, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_c, mid, 3, padding=3, dilation=3, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_c, mid, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(mid * 3, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class ConvNeXtAttUNet3Head(nn.Module):
    """ConvNeXt-Tiny encoder with ASPP, EG-MHSA, and a stride-2 skip stem to retain thin barbs."""

    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        enc = models.convnext_tiny(weights=weights)

        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.fine_stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.enc1 = nn.Sequential(enc.features[0], enc.features[1])
        self.enc2 = nn.Sequential(enc.features[2], enc.features[3])
        self.enc3 = nn.Sequential(enc.features[4], enc.features[5])
        self.enc4 = nn.Sequential(enc.features[6], enc.features[7])

        self.aspp = ASPPBridge(768, 512)
        self.eg_mhsa = EdgeGuidedAttention(channels=512, num_heads=8)

        self.dec4 = ConvBlock(512 + 384, 256)
        self.dec3 = ConvBlock(256 + 192, 128)
        self.dec2 = ConvBlock(128 + 96, 64)
        self.dec1 = ConvBlock(64 + 64, 64)
        self.dec0 = ConvBlock(64 + in_channels, 48)

        self.head_mask = nn.Conv2d(48, 1, kernel_size=1)
        self.head_hull = nn.Conv2d(48, 1, kernel_size=1)
        self.head_spine = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = self.edge_branch(x)
        s0 = self.fine_stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.aspp(e4)
        edge_low = F.interpolate(edge, size=b.shape[2:], mode="bilinear", align_corners=False)
        b = self.eg_mhsa(b, edge_low)

        d4 = self.dec4(torch.cat([F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False), e3], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=e2.shape[2:], mode="bilinear", align_corners=False), e2], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e1.shape[2:], mode="bilinear", align_corners=False), e1], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=s0.shape[2:], mode="bilinear", align_corners=False), s0], dim=1))
        d0 = self.dec0(torch.cat([F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False), x], dim=1))

        return torch.cat([self.head_mask(d0), self.head_hull(d0), self.head_spine(d0)], dim=1)


def multi_head_loss(logits: torch.Tensor, targets: torch.Tensor, tversky_w: float = 0.3) -> torch.Tensor:
    """Weighted BCE + soft Dice + recall-biased Tversky loss across (mask, hull, spine)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)

    weights = torch.tensor([1.0, 0.55, 0.70], device=logits.device).view(1, 3)
    loss = (bce.mean(dim=(2, 3)) * weights).mean() + 0.65 * (dice * weights).mean()

    if tversky_w > 0:
        alpha, beta = 0.35, 0.65
        fp = (probs * (1.0 - targets)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(2, 3))
        tversky = 1.0 - (inter + 1.0) / (inter + alpha * fp + beta * fn + 1.0)
        loss = loss + tversky_w * (tversky * weights).mean()
    return loss


def load_model(weights_path: str, device: str = "cuda") -> nn.Module:
    """Load checkpoint and infer backbone architecture from state_dict keys."""
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if any(k.startswith("enc1.") or k.startswith("aspp.") for k in state):
        model: nn.Module = ConvNeXtAttUNet3Head(in_channels=3, pretrained=False)
    else:
        model = EdgeAttResUNet3Head(in_channels=3, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)
