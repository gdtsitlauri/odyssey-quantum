"""Backend availability detection for the Odyssey quantum track."""

from __future__ import annotations

from dataclasses import asdict, dataclass


try:
    import pennylane as qml  # noqa: F401

    PENNYLANE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PENNYLANE_AVAILABLE = False

try:
    import pennylane_qiskit  # noqa: F401

    PENNYLANE_QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PENNYLANE_QISKIT_AVAILABLE = False

try:
    import qiskit  # noqa: F401

    QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    QISKIT_AVAILABLE = False

try:
    import qiskit_aer  # noqa: F401

    QISKIT_AER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    QISKIT_AER_AVAILABLE = False


@dataclass(frozen=True)
class QuantumBackendAvailability:
    """Installed backend summary for reporting and reproducibility."""

    pennylane_available: bool
    pennylane_qiskit_available: bool
    qiskit_available: bool
    qiskit_aer_available: bool
    preferred_backend: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_backend_availability() -> QuantumBackendAvailability:
    """Return the currently available quantum toolchain summary."""

    preferred_backend = "internal_statevector"
    if PENNYLANE_AVAILABLE and PENNYLANE_QISKIT_AVAILABLE and QISKIT_AER_AVAILABLE:
        preferred_backend = "pennylane+qiskit.aer"
    elif PENNYLANE_AVAILABLE:
        preferred_backend = "pennylane.default.qubit"
    elif QISKIT_AVAILABLE and QISKIT_AER_AVAILABLE:
        preferred_backend = "qiskit.aer"
    elif QISKIT_AVAILABLE:
        preferred_backend = "qiskit"
    return QuantumBackendAvailability(
        pennylane_available=PENNYLANE_AVAILABLE,
        pennylane_qiskit_available=PENNYLANE_QISKIT_AVAILABLE,
        qiskit_available=QISKIT_AVAILABLE,
        qiskit_aer_available=QISKIT_AER_AVAILABLE,
        preferred_backend=preferred_backend,
    )
