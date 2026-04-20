"""Odyssey-Risk model definition and builder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from odyssey.models.baselines import MLPBaseline
from odyssey.models.classical_encoder import ClassicalEncoder
from odyssey.models.quantum_head import (
    PENNYLANE_AVAILABLE,
    QISKIT_AER_AVAILABLE,
    ClassicalUncertaintyHead,
    QuantumHeadStatus,
    QuantumUncertaintyHead,
    RandomUncertaintyHead,
    ZeroUncertaintyHead,
)
from odyssey.models.temporal import GRUBaseline


class OdysseyRiskModel(nn.Module):
    """Hybrid classical-quantum risk combiner."""

    def __init__(self, input_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.encoder = ClassicalEncoder(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("encoder_hidden_dim", 64)),
            latent_dim=int(model_cfg.get("encoder_latent_dim", 24)),
            dropout=float(model_cfg.get("dropout", 0.15)),
        )
        dropout = float(model_cfg.get("dropout", 0.15))
        latent_dim = int(model_cfg.get("encoder_latent_dim", 24))
        attack_hidden_dim = max(16, latent_dim)
        self.linear_probe = nn.Linear(input_dim, 1)
        self.attack_head = nn.Sequential(
            nn.Linear(latent_dim, attack_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attack_hidden_dim, 1),
        )
        self.disable_fragility = bool(config.get("features", {}).get("disable_fragility", False))
        uncertainty_mode = str(model_cfg.get("uncertainty_mode", "auto"))
        quantum_enabled = bool(model_cfg.get("quantum_enabled", True))
        quantum_cfg = model_cfg.get("quantum", {})
        self.quantum_status = self._build_uncertainty_head(
            latent_dim,
            quantum_enabled=quantum_enabled,
            uncertainty_mode=uncertainty_mode,
            require_quantum=bool(model_cfg.get("require_quantum", False)),
            quantum_cfg=quantum_cfg,
        )
        self.uncertainty_gate_net = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        combiner = model_cfg.get("combiner_init", {})
        self.alpha_raw = nn.Parameter(torch.tensor(float(combiner.get("alpha", 1.0))).log())
        self.beta_raw = nn.Parameter(torch.tensor(float(combiner.get("beta", 0.55))).log())
        self.gamma_raw = nn.Parameter(torch.tensor(float(combiner.get("gamma", 0.75))).log())
        self.delta_raw = nn.Parameter(torch.tensor(float(combiner.get("delta", 0.4))).log())
        self.register_buffer("uncertainty_mean", torch.tensor(0.5))
        self.register_buffer("uncertainty_std", torch.tensor(0.25))
        self.register_buffer("fragility_mean", torch.tensor(0.5))
        self.register_buffer("fragility_std", torch.tensor(0.25))

    def _build_uncertainty_head(
        self,
        latent_dim: int,
        quantum_enabled: bool,
        uncertainty_mode: str,
        require_quantum: bool,
        quantum_cfg: dict[str, Any],
    ) -> QuantumHeadStatus:
        backend = str(quantum_cfg.get("backend", "default.qubit"))
        if not quantum_enabled or uncertainty_mode == "zero":
            self.uncertainty_head = ZeroUncertaintyHead()
            return QuantumHeadStatus("zero", backend, "zero", PENNYLANE_AVAILABLE, QISKIT_AER_AVAILABLE)
        if uncertainty_mode == "random":
            self.uncertainty_head = RandomUncertaintyHead()
            return QuantumHeadStatus("random", backend, "random", PENNYLANE_AVAILABLE, QISKIT_AER_AVAILABLE)
        if PENNYLANE_AVAILABLE:
            self.uncertainty_head = QuantumUncertaintyHead(
                input_dim=latent_dim,
                n_qubits=int(quantum_cfg.get("n_qubits", 4)),
                n_layers=int(quantum_cfg.get("n_layers", 2)),
                backend=backend,
                shots=quantum_cfg.get("shots"),
            )
            return QuantumHeadStatus(
                "quantum",
                backend,
                getattr(self.uncertainty_head, "backend_used", backend),
                PENNYLANE_AVAILABLE,
                QISKIT_AER_AVAILABLE,
            )
        if require_quantum:
            raise RuntimeError(
                "Quantum dependencies are required but PennyLane is unavailable. Install optional quantum extras or set model.require_quantum=false."
            )
        self.uncertainty_head = ClassicalUncertaintyHead(latent_dim)
        return QuantumHeadStatus("classical_fallback", backend, "classical_fallback", False, QISKIT_AER_AVAILABLE)

    @staticmethod
    def _positive(parameter: torch.Tensor) -> torch.Tensor:
        return F.softplus(parameter)

    def set_fragility_stats(self, mean: float, std: float) -> None:
        self.fragility_mean.fill_(float(mean))
        self.fragility_std.fill_(max(float(std), 1e-4))

    def set_uncertainty_stats(self, mean: float, std: float) -> None:
        self.uncertainty_mean.fill_(float(mean))
        self.uncertainty_std.fill_(max(float(std), 1e-4))

    def forward(
        self,
        x: torch.Tensor,
        fragility: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.encoder(x)
        attack_logit = self.attack_head(latent).squeeze(-1) + self.linear_probe(x).squeeze(-1)
        attack_prob = torch.sigmoid(attack_logit)
        uncertainty = self.uncertainty_head(latent)
        if fragility is None:
            fragility_score = torch.full_like(uncertainty, float(self.fragility_mean.item()))
        else:
            fragility_score = fragility
        if self.disable_fragility:
            fragility_score = torch.full_like(uncertainty, float(self.fragility_mean.item()))
        standardized_u = ((uncertainty - self.uncertainty_mean) / self.uncertainty_std).clamp(-3.0, 3.0)
        standardized_frag = ((fragility_score - self.fragility_mean) / self.fragility_std).clamp(-3.0, 3.0)
        attack_logit_clamped = torch.logit(attack_prob.clamp(1e-4, 1.0 - 1e-4))
        attack_focus = attack_prob.square()
        gate_inputs = torch.stack([attack_prob, standardized_u, standardized_frag], dim=1)
        uncertainty_gate = torch.sigmoid(self.uncertainty_gate_net(gate_inputs)).squeeze(-1)
        uncertainty_modulation = standardized_u * attack_focus * (0.5 + uncertainty_gate)
        fragility_modulation = standardized_frag * attack_focus * (0.5 + 0.5 * uncertainty_gate)
        interaction = attack_focus * torch.relu(standardized_u) * torch.relu(standardized_frag) * (0.5 + uncertainty_gate)
        risk_logit = (
            self._positive(self.alpha_raw) * attack_logit_clamped
            + self._positive(self.beta_raw) * uncertainty_modulation
            + self._positive(self.gamma_raw) * fragility_modulation
            + self._positive(self.delta_raw) * interaction
        )
        risk_prob = torch.sigmoid(risk_logit)
        return {
            "risk_logit": risk_logit,
            "risk_prob": risk_prob,
            "attack_logit": attack_logit,
            "attack_prob": attack_prob,
            "uncertainty_score": uncertainty,
            "fragility_score": fragility_score,
            "uncertainty_gate": uncertainty_gate,
            "uncertainty_modulation": uncertainty_modulation,
            "fragility_modulation": fragility_modulation,
            "interaction_term": interaction,
        }


def build_model(config: dict[str, Any], feature_spec: dict[str, Any]) -> nn.Module:
    """Build the configured model from feature metadata."""

    input_dim = int(feature_spec["input_dim"])
    model_name = str(config["model"]["name"]).lower()
    if model_name == "odyssey_risk":
        return OdysseyRiskModel(input_dim=input_dim, config=config)
    if model_name == "mlp":
        return MLPBaseline(
            input_dim=input_dim,
            hidden_dim=int(config["model"].get("encoder_hidden_dim", 64)),
            dropout=float(config["model"].get("dropout", 0.15)),
        )
    if model_name == "gru":
        return GRUBaseline(
            input_dim=input_dim,
            hidden_dim=int(config["model"].get("encoder_hidden_dim", 48)),
            dropout=float(config["model"].get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported torch model: {model_name}")

