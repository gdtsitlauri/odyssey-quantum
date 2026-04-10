"""Quantum uncertainty head and fallback variants."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

try:
    import pennylane as qml

    PENNYLANE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    qml = None
    PENNYLANE_AVAILABLE = False

try:
    import pennylane_qiskit  # noqa: F401

    PENNYLANE_QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PENNYLANE_QISKIT_AVAILABLE = False

try:
    import qiskit_aer  # noqa: F401

    QISKIT_AER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    QISKIT_AER_AVAILABLE = False


@dataclass
class QuantumHeadStatus:
    mode: str
    backend_requested: str
    backend_used: str
    quantum_available: bool
    qiskit_aer_available: bool


class ClassicalUncertaintyHead(nn.Module):
    """Deterministic classical fallback for uncertainty estimation."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden_dim = max(8, min(32, input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x).squeeze(-1))


class ZeroUncertaintyHead(nn.Module):
    """Neutral uncertainty output for no-quantum ablations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.size(0),), 0.5, device=x.device, dtype=x.dtype)


class RandomUncertaintyHead(nn.Module):
    """Deterministic random-like ablation head."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = torch.sin(x.sum(dim=1) * 3.17 + 0.618) * 0.5 + 0.5
        return torch.clamp(score, 0.0, 1.0)


class QuantumUncertaintyHead(nn.Module):
    """Small variational quantum circuit head with PennyLane."""

    def __init__(self, input_dim: int, n_qubits: int, n_layers: int, backend: str = "default.qubit", shots: int | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend_requested = backend
        self.backend_used = backend
        self.reducer = nn.Linear(input_dim, n_qubits)
        self.readout = nn.Linear(n_qubits, 1)

        if not PENNYLANE_AVAILABLE:
            raise RuntimeError("PennyLane is required for QuantumUncertaintyHead.")

        resolved_backend = backend
        if backend == "auto":
            if PENNYLANE_QISKIT_AVAILABLE and QISKIT_AER_AVAILABLE:
                resolved_backend = "qiskit.aer"
            else:
                resolved_backend = "default.qubit"
        self.backend_used = resolved_backend

        try:
            device_kwargs = {"wires": n_qubits, "shots": shots}
            if resolved_backend == "qiskit.aer" and shots is None:
                device_kwargs["backend"] = "aer_simulator_statevector"
                self.backend_used = "qiskit.aer[aer_simulator_statevector]"
            self.dev = qml.device(resolved_backend, **device_kwargs)
        except Exception:
            self.backend_used = "default.qubit"
            self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        weights = 0.01 * torch.randn(n_layers, n_qubits, 2)
        self.weights = nn.Parameter(weights)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, circuit_weights: torch.Tensor) -> list[torch.Tensor]:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    qml.RY(circuit_weights[layer, wire, 0], wires=wire)
                    qml.RZ(circuit_weights[layer, wire, 1], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = torch.tanh(self.reducer(x)) * (math.pi / 2.0)
        outputs = []
        for sample in reduced:
            expvals = self.circuit(sample, self.weights)
            if isinstance(expvals, list):
                expvals = torch.stack(expvals)
            outputs.append(expvals.to(dtype=x.dtype))
        stacked = torch.stack(outputs).to(dtype=x.dtype)
        return torch.sigmoid(self.readout(stacked).squeeze(-1))
