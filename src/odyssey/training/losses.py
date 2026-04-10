"""Loss functions for Odyssey training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def focal_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(probs, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha = torch.where(targets > 0.5, torch.full_like(targets, positive_weight), torch.ones_like(targets))
    return (alpha * ((1.0 - pt) ** gamma) * bce).mean()


def brier_like_loss(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean((probs - targets) ** 2)


def temporal_consistency_penalty(
    probs: torch.Tensor,
    targets: torch.Tensor,
    sequence_ids: torch.Tensor,
    timestamps: torch.Tensor,
    max_gap: float = 5.0,
) -> torch.Tensor:
    if probs.numel() < 2:
        return torch.zeros((), device=probs.device)
    combined_key = sequence_ids.float() * (timestamps.max().detach() + 1.0) + timestamps
    order = torch.argsort(combined_key)
    sorted_probs = probs[order]
    sorted_targets = targets[order]
    sorted_seq = sequence_ids[order]
    sorted_time = timestamps[order]
    mask = (
        (sorted_seq[1:] == sorted_seq[:-1])
        & (sorted_targets[1:] == sorted_targets[:-1])
        & ((sorted_time[1:] - sorted_time[:-1]).abs() <= max_gap)
    )
    if not mask.any():
        return torch.zeros((), device=probs.device)
    deltas = sorted_probs[1:] - sorted_probs[:-1]
    return torch.mean((deltas[mask]) ** 2)


def minority_attack_margin_term(
    probs: torch.Tensor,
    targets: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    attack_mask = targets > 0.5
    benign_mask = ~attack_mask
    if attack_mask.sum() == 0 or benign_mask.sum() == 0:
        return torch.zeros((), device=probs.device)
    attack_mean = probs[attack_mask].mean()
    benign_mean = probs[benign_mask].mean()
    return torch.relu(torch.tensor(margin, device=probs.device) - (attack_mean - benign_mean))


def composite_odyssey_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    sequence_ids: torch.Tensor,
    timestamps: torch.Tensor,
    training_cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    risk_probs = outputs["risk_prob"]
    attack_logits = outputs.get("attack_logit")
    if attack_logits is None:
        attack_logits = torch.logit(risk_probs.clamp(1e-6, 1.0 - 1e-6))
    attack_aux = F.binary_cross_entropy_with_logits(attack_logits, targets)
    focal = focal_loss(
        risk_probs,
        targets,
        gamma=float(training_cfg.get("focal_gamma", 2.0)),
        positive_weight=float(training_cfg.get("positive_class_weight", 1.0)),
    )
    brier = brier_like_loss(risk_probs, targets)
    temporal = temporal_consistency_penalty(risk_probs, targets, sequence_ids, timestamps)
    margin = minority_attack_margin_term(
        risk_probs,
        targets,
        margin=float(training_cfg.get("minority_margin", 0.08)),
    )
    total = (
        focal
        + float(training_cfg.get("lambda_attack", 0.0)) * attack_aux
        + float(training_cfg.get("lambda_brier", 0.0)) * brier
        + float(training_cfg.get("lambda_temp", 0.0)) * temporal
        + float(training_cfg.get("lambda_margin", 0.0)) * margin
    )
    return total, {
        "attack_aux": float(attack_aux.detach().cpu()),
        "focal": float(focal.detach().cpu()),
        "brier": float(brier.detach().cpu()),
        "temporal": float(temporal.detach().cpu()),
        "margin": float(margin.detach().cpu()),
    }

