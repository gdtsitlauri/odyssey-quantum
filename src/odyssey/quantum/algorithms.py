"""Toy quantum algorithm suite for Odyssey."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from odyssey.quantum.simulator import (
    H,
    apply_cnot,
    apply_single_qubit_gate,
    basis_state,
    expectation_hamiltonian,
    labeled_probabilities,
    marginal_probabilities,
    pauli_string_operator,
    rx,
    ry,
    top_measurements,
    zero_state,
)


def _bits_to_int(bits: str) -> int:
    return int(bits, 2)


def _boolean_oracle_matrix(n_inputs: int, function: Callable[[str], int]) -> np.ndarray:
    n_qubits = n_inputs + 1
    dimension = 2**n_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for index in range(dimension):
        bits = format(index, f"0{n_qubits}b")
        x_bits = bits[:n_inputs]
        ancilla = int(bits[-1])
        output_bits = f"{x_bits}{ancilla ^ int(function(x_bits))}"
        matrix[_bits_to_int(output_bits), index] = 1.0
    return matrix


def _apply_hadamards(state: np.ndarray, wires: list[int], n_qubits: int) -> np.ndarray:
    updated = state
    for wire in wires:
        updated = apply_single_qubit_gate(updated, H, wire=wire, n_qubits=n_qubits)
    return updated


def _run_deutsch_jozsa_case(oracle_name: str, function: Callable[[str], int], n_inputs: int) -> dict[str, Any]:
    n_qubits = n_inputs + 1
    state = basis_state("0" * n_inputs + "1")
    state = _apply_hadamards(state, list(range(n_qubits)), n_qubits)
    oracle = _boolean_oracle_matrix(n_inputs, function)
    state = oracle @ state
    state = _apply_hadamards(state, list(range(n_inputs)), n_qubits)
    marginal = marginal_probabilities(state, keep_wires=list(range(n_inputs)), n_qubits=n_qubits)
    measured = max(marginal, key=marginal.get)
    predicted = "constant" if measured == "0" * n_inputs else "balanced"
    return {
        "algorithm": "deutsch_jozsa",
        "oracle": oracle_name,
        "predicted_class": predicted,
        "correct": predicted == oracle_name,
        "all_zero_probability": float(marginal["0" * n_inputs]),
        "dominant_measurement": measured,
    }


def _run_bernstein_vazirani(hidden_string: str) -> dict[str, Any]:
    n_inputs = len(hidden_string)

    def oracle_function(x_bits: str) -> int:
        return sum(int(a) * int(b) for a, b in zip(hidden_string, x_bits, strict=True)) % 2

    n_qubits = n_inputs + 1
    state = basis_state("0" * n_inputs + "1")
    state = _apply_hadamards(state, list(range(n_qubits)), n_qubits)
    state = _boolean_oracle_matrix(n_inputs, oracle_function) @ state
    state = _apply_hadamards(state, list(range(n_inputs)), n_qubits)
    marginal = marginal_probabilities(state, keep_wires=list(range(n_inputs)), n_qubits=n_qubits)
    recovered = max(marginal, key=marginal.get)
    return {
        "algorithm": "bernstein_vazirani",
        "hidden_string": hidden_string,
        "recovered_string": recovered,
        "success_probability": float(marginal[hidden_string]),
        "correct": recovered == hidden_string,
    }


def _grover_oracle(marked_index: int, dimension: int) -> np.ndarray:
    oracle = np.eye(dimension, dtype=np.complex128)
    oracle[marked_index, marked_index] = -1.0
    return oracle


def _grover_diffusion(dimension: int) -> np.ndarray:
    uniform = np.full((dimension, 1), 1.0 / math.sqrt(dimension), dtype=np.complex128)
    return 2.0 * (uniform @ uniform.T.conj()) - np.eye(dimension, dtype=np.complex128)


def _run_grover(n_qubits: int, marked_bitstring: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dimension = 2**n_qubits
    marked_index = _bits_to_int(marked_bitstring)
    state = zero_state(n_qubits)
    state = _apply_hadamards(state, list(range(n_qubits)), n_qubits)
    oracle = _grover_oracle(marked_index, dimension)
    diffusion = _grover_diffusion(dimension)
    recommended_iterations = max(1, int(round((math.pi / 4.0) * math.sqrt(dimension))))
    trace_rows = []
    best_probability = -1.0
    best_state = None
    for iteration in range(recommended_iterations + 1):
        probabilities = labeled_probabilities(state, n_qubits)
        success_probability = float(probabilities[marked_bitstring])
        trace_rows.append(
            {
                "iteration": iteration,
                "marked_state": marked_bitstring,
                "success_probability": success_probability,
                "dominant_measurement": max(probabilities, key=probabilities.get),
            }
        )
        if success_probability > best_probability:
            best_probability = success_probability
            best_state = max(probabilities, key=probabilities.get)
        state = diffusion @ (oracle @ state)
    summary = {
        "algorithm": "grover",
        "n_qubits": n_qubits,
        "marked_state": marked_bitstring,
        "recommended_iterations": recommended_iterations,
        "best_success_probability": best_probability,
        "best_measurement": best_state,
        "correct": best_state == marked_bitstring,
    }
    return summary, trace_rows


VQE_HAMILTONIAN = [
    (-1.052373245772859, "II"),
    (0.39793742484318045, "ZI"),
    (-0.39793742484318045, "IZ"),
    (-0.01128010425623538, "ZZ"),
    (0.18093119978423156, "XX"),
]


def _vqe_state(params: np.ndarray) -> np.ndarray:
    if params.size < 6:
        params = np.pad(params, (0, 6 - params.size))
    state = zero_state(2)
    state = apply_single_qubit_gate(state, ry(float(params[0])), wire=0, n_qubits=2)
    state = apply_single_qubit_gate(state, ry(float(params[1])), wire=1, n_qubits=2)
    state = apply_cnot(state, control=0, target=1, n_qubits=2)
    state = apply_single_qubit_gate(state, ry(float(params[2])), wire=0, n_qubits=2)
    state = apply_single_qubit_gate(state, ry(float(params[3])), wire=1, n_qubits=2)
    state = apply_cnot(state, control=1, target=0, n_qubits=2)
    state = apply_single_qubit_gate(state, rx(float(params[4])), wire=0, n_qubits=2)
    state = apply_single_qubit_gate(state, rx(float(params[5])), wire=1, n_qubits=2)
    return state


def _run_vqe(initial_params: list[float], maxiter: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    operator = np.zeros((4, 4), dtype=np.complex128)
    for coefficient, pauli in VQE_HAMILTONIAN:
        operator = operator + coefficient * pauli_string_operator(pauli)
    exact_ground_energy = float(np.min(np.linalg.eigvalsh(operator)).real)
    trace_rows: list[dict[str, Any]] = []

    def objective(params: np.ndarray) -> float:
        state = _vqe_state(params)
        energy = expectation_hamiltonian(state, VQE_HAMILTONIAN)
        trace_rows.append(
            {
                "step": len(trace_rows),
                "energy": float(energy),
                "theta0": float(params[0]),
                "theta1": float(params[1]),
                "theta2": float(params[2]),
                "theta3": float(params[3]),
                "theta4": float(params[4]) if params.size > 4 else 0.0,
                "theta5": float(params[5]) if params.size > 5 else 0.0,
            }
        )
        return float(energy)

    seeds = [
        np.asarray(initial_params, dtype=float),
        np.zeros(6, dtype=float),
        np.asarray([0.0, math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0], dtype=float),
    ]
    start = time.perf_counter()
    best_result = None
    for seed in seeds:
        result = minimize(
            objective,
            seed,
            method="Powell",
            options={"maxiter": int(maxiter), "xtol": 1e-4, "ftol": 1e-4},
        )
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result
    runtime_seconds = time.perf_counter() - start
    assert best_result is not None
    best_energy = float(best_result.fun)
    return (
        {
            "algorithm": "vqe",
            "ground_energy_estimate": best_energy,
            "exact_ground_energy": exact_ground_energy,
            "absolute_error": abs(best_energy - exact_ground_energy),
            "iterations": int(best_result.nit),
            "runtime_seconds": runtime_seconds,
            "converged": bool(best_result.success),
        },
        trace_rows,
    )


def _qaoa_cost_unitary(state: np.ndarray, gamma: float, edges: list[tuple[int, int]], n_qubits: int) -> np.ndarray:
    phased = state.copy()
    for index in range(2**n_qubits):
        bits = format(index, f"0{n_qubits}b")
        cost = 0.0
        for left, right in edges:
            if bits[left] != bits[right]:
                cost += 1.0
        phased[index] *= np.exp(-1j * gamma * cost)
    return phased


def _qaoa_expected_cut(state: np.ndarray, edges: list[tuple[int, int]], n_qubits: int) -> float:
    probabilities = labeled_probabilities(state, n_qubits)
    expectation = 0.0
    for bitstring, probability in probabilities.items():
        cost = sum(1 for left, right in edges if bitstring[left] != bitstring[right])
        expectation += probability * cost
    return float(expectation)


def _run_qaoa(gamma_steps: int, beta_steps: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_qubits = 3
    edges = [(0, 1), (1, 2), (0, 2)]
    optimal_cut = 2.0
    best_row: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    gammas = np.linspace(0.0, math.pi, gamma_steps)
    betas = np.linspace(0.0, math.pi / 2.0, beta_steps)
    for gamma in gammas:
        for beta in betas:
            state = zero_state(n_qubits)
            state = _apply_hadamards(state, list(range(n_qubits)), n_qubits)
            state = _qaoa_cost_unitary(state, float(gamma), edges, n_qubits)
            for wire in range(n_qubits):
                state = apply_single_qubit_gate(state, rx(2.0 * float(beta)), wire=wire, n_qubits=n_qubits)
            expectation = _qaoa_expected_cut(state, edges, n_qubits)
            probabilities = labeled_probabilities(state, n_qubits)
            top_state, top_probability = top_measurements(probabilities, k=1)[0]
            row = {
                "gamma": float(gamma),
                "beta": float(beta),
                "expected_cut": float(expectation),
                "approximation_ratio": float(expectation / optimal_cut),
                "top_measurement": top_state,
                "top_probability": float(top_probability),
            }
            rows.append(row)
            if best_row is None or row["expected_cut"] > best_row["expected_cut"]:
                best_row = row
    assert best_row is not None
    summary = {
        "algorithm": "qaoa",
        "graph": "triangle_maxcut",
        "best_expected_cut": best_row["expected_cut"],
        "approximation_ratio": best_row["approximation_ratio"],
        "best_gamma": best_row["gamma"],
        "best_beta": best_row["beta"],
        "dominant_measurement": best_row["top_measurement"],
    }
    return summary, rows


def _multiplicative_order(base: int, modulus: int) -> int:
    value = 1
    for order in range(1, modulus + 1):
        value = (value * base) % modulus
        if value == 1:
            return order
    raise ValueError("Multiplicative order not found.")


def _run_shor_toy(composite: int, base: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sequence_rows = []
    value = 1
    for exponent in range(0, composite):
        sequence_rows.append(
            {
                "exponent": exponent,
                "value": value,
            }
        )
        value = (value * base) % composite
    order = _multiplicative_order(base, composite)
    half_power = pow(base, order // 2, composite)
    factor_left = math.gcd(half_power - 1, composite)
    factor_right = math.gcd(half_power + 1, composite)
    summary = {
        "algorithm": "shor_toy_reference",
        "composite": composite,
        "base": base,
        "order": order,
        "half_power_mod_n": half_power,
        "factor_left": int(factor_left),
        "factor_right": int(factor_right),
        "successful_factorization": factor_left not in {1, composite} and factor_right not in {1, composite},
        "note": "Arithmetic/order-finding walk-through, not a full fault-tolerant Shor circuit.",
    }
    return summary, sequence_rows


def run_algorithm_suite(config: dict[str, Any]) -> dict[str, Any]:
    """Run a deterministic educational quantum algorithm suite."""

    algorithms_cfg = config.get("quantum", {}).get("algorithms", {})
    dj_qubits = int(algorithms_cfg.get("deutsch_jozsa", {}).get("n_input_qubits", 2))
    bv_hidden = str(algorithms_cfg.get("bernstein_vazirani", {}).get("hidden_string", "101"))
    grover_cfg = algorithms_cfg.get("grover", {})
    grover_qubits = int(grover_cfg.get("n_qubits", 3))
    grover_marked = str(grover_cfg.get("marked_bitstring", "101"))
    vqe_cfg = algorithms_cfg.get("vqe", {})
    qaoa_cfg = algorithms_cfg.get("qaoa", {})
    shor_cfg = algorithms_cfg.get("shor", {})

    dj_cases = [
        _run_deutsch_jozsa_case("constant", lambda _bits: 0, dj_qubits),
        _run_deutsch_jozsa_case("balanced", lambda bits: sum(int(bit) for bit in bits) % 2, dj_qubits),
    ]
    bv_summary = _run_bernstein_vazirani(bv_hidden)
    grover_summary, grover_rows = _run_grover(grover_qubits, grover_marked)
    vqe_summary, vqe_rows = _run_vqe(
        initial_params=list(vqe_cfg.get("initial_params", [0.2, -0.3, 0.1, 0.05, 0.0, 0.0])),
        maxiter=int(vqe_cfg.get("maxiter", 80)),
    )
    qaoa_summary, qaoa_rows = _run_qaoa(
        gamma_steps=int(qaoa_cfg.get("gamma_steps", 21)),
        beta_steps=int(qaoa_cfg.get("beta_steps", 21)),
    )
    shor_summary, shor_rows = _run_shor_toy(
        composite=int(shor_cfg.get("composite", 15)),
        base=int(shor_cfg.get("base", 2)),
    )

    summary_rows = dj_cases + [bv_summary, grover_summary, vqe_summary, qaoa_summary, shor_summary]
    return {
        "summary": {
            "suite": "quantum_algorithms",
            "algorithms": summary_rows,
        },
        "tables": {
            "summary": pd.DataFrame(summary_rows),
            "grover_iterations": pd.DataFrame(grover_rows),
            "vqe_trace": pd.DataFrame(vqe_rows),
            "qaoa_grid": pd.DataFrame(qaoa_rows),
            "shor_sequence": pd.DataFrame(shor_rows),
        },
    }
