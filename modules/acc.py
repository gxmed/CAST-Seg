"""Agreement-aware Confidence Calibration (ACC)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class AgreementAwareConfidenceCalibration(nn.Module):
    """Calibrate foreground/background decoder outputs as defined in the paper.

    ``foreground_prob`` is :math:`P_f`; ``background_prob`` is :math:`P_b`.
    The complementary foreground candidate is therefore ``1 - P_b``.
    """

    def __init__(
        self,
        decision_threshold: float = 0.5,
        agreement_threshold: float = 0.30,
        conflict_threshold: float = 0.70,
        confidence_margin: float = 0.20,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < decision_threshold < 1.0:
            raise ValueError("decision_threshold must be in (0, 1).")
        self.decision_threshold = float(decision_threshold)
        self.agreement_threshold = float(agreement_threshold)
        self.conflict_threshold = float(conflict_threshold)
        self.confidence_margin = float(confidence_margin)
        self.eps = float(eps)

    def _confidence(self, probability: torch.Tensor) -> torch.Tensor:
        probability = probability.detach().float().clamp(self.eps, 1.0 - self.eps)
        # Equation (ACC): |P - tau| / (1 - tau). The published experiments use tau=0.5.
        return torch.abs(probability - self.decision_threshold) / (1.0 - self.decision_threshold)

    def calibrate_pair(
        self,
        probability_a: torch.Tensor,
        probability_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        probability_a = probability_a.detach().float().clamp(self.eps, 1.0 - self.eps)
        probability_b = probability_b.detach().float().clamp(self.eps, 1.0 - self.eps)

        confidence_a = self._confidence(probability_a)
        confidence_b = self._confidence(probability_b)
        label_a = probability_a > self.decision_threshold
        label_b = probability_b > self.decision_threshold
        agreement = label_a == label_b

        max_confidence = torch.maximum(confidence_a, confidence_b)
        confidence_gap = torch.abs(confidence_a - confidence_b)
        fused_probability = (
            confidence_a * probability_a + confidence_b * probability_b
        ) / (confidence_a + confidence_b + self.eps)
        selected_probability = torch.where(
            confidence_a >= confidence_b,
            probability_a,
            probability_b,
        )
        pseudo_probability = torch.where(agreement, fused_probability, selected_probability)
        pseudo_label = (pseudo_probability > self.decision_threshold).float()

        reliable_agreement = agreement & (max_confidence >= self.agreement_threshold)
        reliable_conflict = (
            (~agreement)
            & (max_confidence >= self.conflict_threshold)
            & (confidence_gap >= self.confidence_margin)
        )
        reliable_mask = (reliable_agreement | reliable_conflict).float()
        debug = {
            "pseudo_probability": pseudo_probability,
            "reliable_mask": reliable_mask,
            "reliable_ratio": reliable_mask.mean().detach(),
        }
        return pseudo_label, reliable_mask, debug

    def forward(self, foreground_prob: torch.Tensor, background_prob: torch.Tensor):
        pseudo_fg, mask_fg, debug_fg = self.calibrate_pair(
            foreground_prob,
            1.0 - background_prob,
        )
        pseudo_bg, mask_bg, debug_bg = self.calibrate_pair(
            background_prob,
            1.0 - foreground_prob,
        )
        return pseudo_fg, pseudo_bg, mask_fg, mask_bg, {"fg": debug_fg, "bg": debug_bg}
