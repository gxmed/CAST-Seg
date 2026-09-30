"""Adaptive Semantic Refinement (ASR) scheduling utilities."""

from __future__ import annotations


class AdaptiveSemanticRefinementSchedule:
    """Periodic CAM-prior refresh schedule used during semi-supervised training."""

    def __init__(self, warmup_epochs: int = 5, update_interval: int = 1) -> None:
        if update_interval < 1:
            raise ValueError("update_interval must be at least 1.")
        self.warmup_epochs = int(warmup_epochs)
        self.update_interval = int(update_interval)

    def should_update(self, epoch: int) -> bool:
        epoch = int(epoch)
        return epoch >= self.warmup_epochs and (
            (epoch - self.warmup_epochs) % self.update_interval == 0
        )
