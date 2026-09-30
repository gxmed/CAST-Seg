"""Medical Domain Adaptation Adapter (MDAA) building blocks.

These blocks are inserted into the Medical-SAM3 ViT/prompt encoder during the
separate expert-adaptation stage. The resulting expert masks are consumed by
the CAST-Seg semi-supervised stage; Medical-SAM3 is not required at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MedicalDomainAdaptationAdapter(nn.Module):
    """Semantic bottleneck adapter used in deeper transformer blocks."""

    def __init__(self, feature_dim: int, mlp_ratio: float = 0.25, scale: float = 1.0):
        super().__init__()
        hidden_dim = int(feature_dim * mlp_ratio)
        self.down_projection = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.GELU()
        self.up_projection = nn.Linear(hidden_dim, feature_dim)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_projection(self.activation(self.down_projection(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.residual(x)


class SpatialAwareAdapter(MedicalDomainAdaptationAdapter):
    """Shallow-layer MDAA with depth-wise 3x3 spatial convolution for BHWC features."""

    def __init__(self, feature_dim: int, mlp_ratio: float = 0.25, scale: float = 1.0):
        super().__init__(feature_dim, mlp_ratio=mlp_ratio, scale=scale)
        hidden_dim = int(feature_dim * mlp_ratio)
        self.depthwise_convolution = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SpatialAwareAdapter expects BHWC features, got {tuple(x.shape)}")
        hidden = self.down_projection(x).permute(0, 3, 1, 2).contiguous()
        hidden = self.depthwise_convolution(hidden).permute(0, 2, 3, 1).contiguous()
        return self.up_projection(self.activation(hidden))
