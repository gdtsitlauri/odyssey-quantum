from __future__ import annotations

import torch

from odyssey.training.losses import (
    composite_odyssey_loss,
    minority_attack_margin_term,
    temporal_consistency_penalty,
)


def test_composite_loss_is_finite() -> None:
    outputs = {"risk_prob": torch.tensor([0.2, 0.8, 0.6, 0.1], dtype=torch.float32)}
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    sequence_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    timestamps = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    loss, components = composite_odyssey_loss(
        outputs,
        targets,
        sequence_ids,
        timestamps,
        {
            "focal_gamma": 2.0,
            "positive_class_weight": 1.2,
            "lambda_brier": 0.2,
            "lambda_temp": 0.1,
            "lambda_margin": 0.1,
            "minority_margin": 0.05,
        },
    )
    assert torch.isfinite(loss)
    assert {"focal", "brier", "temporal", "margin"} <= set(components)


def test_temporal_penalty_and_margin_term_non_negative() -> None:
    probs = torch.tensor([0.1, 0.2, 0.8, 0.85], dtype=torch.float32)
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    sequence_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    timestamps = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    temporal = temporal_consistency_penalty(probs, targets, sequence_ids, timestamps)
    margin = minority_attack_margin_term(probs, targets, margin=0.05)
    assert temporal >= 0.0
    assert margin >= 0.0


