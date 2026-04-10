"""Temporal models for sequence mode."""

from __future__ import annotations

import torch
from torch import nn


class GRUBaseline(nn.Module):
    """Simple GRU-based temporal baseline."""

    def __init__(self, input_dim: int, hidden_dim: int = 48, num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        fragility: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if lengths is None:
            outputs, hidden = self.gru(x)
            final_hidden = hidden[-1]
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            final_hidden = hidden[-1]
        logits = self.classifier(final_hidden).squeeze(-1)
        probs = torch.sigmoid(logits)
        return {"risk_logit": logits, "risk_prob": probs, "attack_logit": logits, "attack_prob": probs}

