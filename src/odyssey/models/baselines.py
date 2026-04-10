"""Baseline models and sklearn estimator factories."""

from __future__ import annotations

from typing import Any

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn


def make_sklearn_estimator(name: str, seed: int) -> Any:
    """Create a classical baseline estimator."""

    if name == "logistic_regression":
        return LogisticRegression(
            random_state=seed,
            max_iter=400,
            class_weight="balanced",
            solver="liblinear",
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=180,
            max_depth=14,
            min_samples_leaf=2,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
    raise ValueError(f"Unsupported sklearn baseline: {name}")


class MLPBaseline(nn.Module):
    """Lightweight tabular MLP baseline."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        fragility: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.network(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return {"risk_logit": logits, "risk_prob": probs, "attack_logit": logits, "attack_prob": probs}
