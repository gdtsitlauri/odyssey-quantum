"""Quantum foundations demos for the Odyssey quantum track."""

from __future__ import annotations

from typing import Any

import pandas as pd

from odyssey.quantum.availability import detect_backend_availability
from odyssey.quantum.simulator import (
    H,
    apply_cnot,
    apply_single_qubit_gate,
    classical_fidelity,
    labeled_probabilities,
    marginal_probabilities,
    measurement_probabilities,
    shannon_entropy,
    uniform_mixture,
    zero_state,
)


def _superposition_demo() -> dict[str, Any]:
    state = zero_state(1)
    state = apply_single_qubit_gate(state, H, wire=0, n_qubits=1)
    probabilities = labeled_probabilities(state, 1)
    return {
        "experiment": "single_qubit_superposition",
        "dominant_state": max(probabilities, key=probabilities.get),
        "entropy_bits": shannon_entropy(list(probabilities.values())),
        "probabilities": probabilities,
    }


def _bell_demo() -> dict[str, Any]:
    state = zero_state(2)
    state = apply_single_qubit_gate(state, H, wire=0, n_qubits=2)
    state = apply_cnot(state, control=0, target=1, n_qubits=2)
    full_probabilities = labeled_probabilities(state, 2)
    marginal_q0 = marginal_probabilities(state, keep_wires=[0], n_qubits=2)
    marginal_q1 = marginal_probabilities(state, keep_wires=[1], n_qubits=2)
    parity_correlation = full_probabilities["00"] + full_probabilities["11"]
    return {
        "experiment": "bell_state",
        "parity_correlation": parity_correlation,
        "marginal_q0_entropy": shannon_entropy(list(marginal_q0.values())),
        "marginal_q1_entropy": shannon_entropy(list(marginal_q1.values())),
        "probabilities": full_probabilities,
    }


def _ghz_demo(n_qubits: int) -> dict[str, Any]:
    state = zero_state(n_qubits)
    state = apply_single_qubit_gate(state, H, wire=0, n_qubits=n_qubits)
    for target in range(1, n_qubits):
        state = apply_cnot(state, control=0, target=target, n_qubits=n_qubits)
    probabilities = labeled_probabilities(state, n_qubits)
    dominant_states = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:2]
    return {
        "experiment": f"ghz_{n_qubits}q",
        "dominant_states": dominant_states,
        "entropy_bits": shannon_entropy(list(probabilities.values())),
        "probabilities": probabilities,
    }


def _noise_scan(noise_levels: list[float]) -> list[dict[str, float]]:
    state = zero_state(2)
    state = apply_single_qubit_gate(state, H, wire=0, n_qubits=2)
    state = apply_cnot(state, control=0, target=1, n_qubits=2)
    ideal = measurement_probabilities(state)
    rows: list[dict[str, float]] = []
    for noise in noise_levels:
        noisy = uniform_mixture(ideal, mixing=noise)
        rows.append(
            {
                "depolarizing_mixture": float(noise),
                "bell_fidelity": classical_fidelity(ideal, noisy),
                "entropy_bits": shannon_entropy(noisy),
                "prob_00": float(noisy[0]),
                "prob_11": float(noisy[3]),
            }
        )
    return rows


def run_foundations_suite(config: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic quantum foundations demos."""

    foundations_cfg = config.get("quantum", {}).get("foundations", {})
    noise_levels = list(foundations_cfg.get("noise_levels", [0.0, 0.05, 0.1, 0.2, 0.35]))
    ghz_qubits = int(foundations_cfg.get("ghz_qubits", 3))

    backend_status = detect_backend_availability()
    experiments = [_superposition_demo(), _bell_demo(), _ghz_demo(ghz_qubits)]
    state_rows = []
    for experiment in experiments:
        probabilities = experiment["probabilities"]
        top_state, top_probability = max(probabilities.items(), key=lambda item: item[1])
        state_rows.append(
            {
                "experiment": experiment["experiment"],
                "top_state": top_state,
                "top_probability": float(top_probability),
                "support_size": int(sum(1 for value in probabilities.values() if value > 1e-9)),
                "entropy_bits": float(
                    experiment.get("entropy_bits")
                    or experiment.get("marginal_q0_entropy")
                    or experiment.get("marginal_q1_entropy")
                    or 0.0
                ),
            }
        )

    noise_rows = _noise_scan(noise_levels)
    summary = {
        "suite": "quantum_foundations",
        "backend_status": backend_status.to_dict(),
        "state_preparation": state_rows,
        "noise_scan": noise_rows,
    }
    return {
        "summary": summary,
        "tables": {
            "state_preparation": pd.DataFrame(state_rows),
            "noise_scan": pd.DataFrame(noise_rows),
        },
        "experiments": experiments,
    }
