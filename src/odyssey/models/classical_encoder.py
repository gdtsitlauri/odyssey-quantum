"""Classical encoder building blocks."""

from __future__ import annotations

import torch
from torch import nn


class ClassicalEncoder(nn.Module):
    """Lightweight MLP encoder used by Odyssey."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


