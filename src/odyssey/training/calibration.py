"""Post-hoc calibration for neural baselines."""

from __future__ import annotations

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling on logits."""

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-4)

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50) -> "TemperatureScaler":
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

