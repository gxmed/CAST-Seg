"""Weighted Exponential Moving Average (WEMA)."""

from __future__ import annotations

import torch


@torch.no_grad()
def weighted_ema_update(
    teacher,
    student_a,
    student_b,
    beta: float = 0.99,
    student_a_weight: float = 0.75,
) -> None:
    """Apply the two equations from the WEMA subsection in-place.

    ``student_a_weight`` is the paper's lambda and ``beta`` is the EMA momentum.
    """

    beta = float(beta)
    student_a_weight = float(student_a_weight)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    if not 0.0 <= student_a_weight <= 1.0:
        raise ValueError("student_a_weight must be in [0, 1].")
    student_b_weight = 1.0 - student_a_weight

    for teacher_parameter, parameter_a, parameter_b in zip(
        teacher.parameters(),
        student_a.parameters(),
        student_b.parameters(),
    ):
        mixed_student = (
            student_a_weight * parameter_a.data
            + student_b_weight * parameter_b.data
        )
        teacher_parameter.data.mul_(beta).add_(mixed_student, alpha=1.0 - beta)
