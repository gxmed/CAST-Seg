"""Synchronized Cross-modal Augmentation (SCA)."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SynchronizedCrossModalAugmentation:
    """Apply identical geometric transforms to image, mask, CAM and SAM prior."""

    def __init__(
        self,
        probability: float = 0.5,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        rotate: bool = True,
        max_angle: float = 10.0,
    ) -> None:
        self.probability = float(probability)
        self.horizontal_flip = bool(horizontal_flip)
        self.vertical_flip = bool(vertical_flip)
        self.rotate = bool(rotate)
        self.max_angle = float(max_angle)

    @staticmethod
    def _rotate_tensor(x: Optional[torch.Tensor], angle_deg: float, mode: str):
        if x is None:
            return None
        original_dtype = x.dtype
        x_float = x.float().unsqueeze(0)
        angle = float(angle_deg) * np.pi / 180.0
        cosine, sine = np.cos(angle), np.sin(angle)
        theta = torch.tensor(
            [[[cosine, -sine, 0.0], [sine, cosine, 0.0]]],
            dtype=x_float.dtype,
            device=x_float.device,
        )
        grid = F.affine_grid(theta, size=x_float.shape, align_corners=False)
        output = F.grid_sample(
            x_float,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        )[0]
        if original_dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
        ):
            return output.round().to(original_dtype)
        return output.to(original_dtype)

    def __call__(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        cam: Optional[torch.Tensor] = None,
        sam: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        fields = [image, label, cam, sam]
        if self.horizontal_flip and random.random() < self.probability:
            fields = [None if value is None else torch.flip(value, dims=[-1]) for value in fields]
        if self.vertical_flip and random.random() < self.probability:
            fields = [None if value is None else torch.flip(value, dims=[-2]) for value in fields]
        if self.rotate and self.max_angle > 0 and random.random() < self.probability:
            angle = random.uniform(-self.max_angle, self.max_angle)
            fields = [
                self._rotate_tensor(fields[0], angle, "bilinear"),
                self._rotate_tensor(fields[1].float(), angle, "nearest").to(label.dtype),
                self._rotate_tensor(fields[2], angle, "bilinear"),
                self._rotate_tensor(fields[3], angle, "bilinear"),
            ]
        return tuple(None if value is None else value.contiguous() for value in fields)
