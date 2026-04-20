"""Small exact statevector utilities for toy quantum experiments."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np


I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def rx(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [
            [np.cos(half), -1j * np.sin(half)],
            [-1j * np.sin(half), np.cos(half)],
        ],
        dtype=np.complex128,
    )


def ry(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [
            [np.cos(half), -np.sin(half)],
            [np.sin(half), np.cos(half)],
        ],
        dtype=np.complex128,
    )


def rz(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [
            [np.exp(-1j * half), 0.0],
            [0.0, np.exp(1j * half)],
        ],
        dtype=np.complex128,
    )


def bitstrings(n_qubits: int) -> list[str]:
    return ["".join(bits) for bits in itertools.product("01", repeat=n_qubits)]


def zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    return state


def basis_state(bitstring: str) -> np.ndarray:
    state = np.zeros(2**len(bitstring), dtype=np.complex128)
    state[int(bitstring, 2)] = 1.0
    return state


def kron_all(operators: Sequence[np.ndarray]) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def single_qubit_operator(gate: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
    operators = [I] * n_qubits
    operators[wire] = gate
    return kron_all(operators)


def apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
    return single_qubit_operator(gate, wire, n_qubits) @ state


def cnot_operator(control: int, target: int, n_qubits: int) -> np.ndarray:
    dimension = 2**n_qubits
    operator = np.zeros((dimension, dimension), dtype=np.complex128)
    for index in range(dimension):
        bits = list(format(index, f"0{n_qubits}b"))
        if bits[control] == "1":
            bits[target] = "0" if bits[target] == "1" else "1"
        target_index = int("".join(bits), 2)
        operator[target_index, index] = 1.0
    return operator


def apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    return cnot_operator(control, target, n_qubits) @ state


def measurement_probabilities(state: np.ndarray) -> np.ndarray:
    probabilities = np.abs(state) ** 2
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("State probabilities sum to zero.")
    return probabilities / total


def labeled_probabilities(state: np.ndarray, n_qubits: int) -> dict[str, float]:
    probabilities = measurement_probabilities(state)
    return {label: float(probabilities[index]) for index, label in enumerate(bitstrings(n_qubits))}


def marginal_probabilities(state: np.ndarray, keep_wires: Iterable[int], n_qubits: int) -> dict[str, float]:
    keep = list(keep_wires)
    probabilities = measurement_probabilities(state)
    aggregated = {label: 0.0 for label in bitstrings(len(keep))}
    for index, probability in enumerate(probabilities):
        bits = format(index, f"0{n_qubits}b")
        key = "".join(bits[wire] for wire in keep)
        aggregated[key] += float(probability)
    return aggregated


def shannon_entropy(probabilities: Sequence[float], eps: float = 1e-12) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > eps]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def classical_fidelity(probabilities_a: Sequence[float], probabilities_b: Sequence[float]) -> float:
    a = np.asarray(probabilities_a, dtype=float)
    b = np.asarray(probabilities_b, dtype=float)
    return float(np.square(np.sum(np.sqrt(np.clip(a * b, 0.0, None)))))


def uniform_mixture(probabilities: Sequence[float], mixing: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    dimension = probs.size
    uniform = np.full(dimension, 1.0 / dimension, dtype=float)
    return (1.0 - mixing) * probs + mixing * uniform


def pauli_string_operator(pauli_string: str) -> np.ndarray:
    mapping = {"I": I, "X": X, "Y": Y, "Z": Z}
    return kron_all([mapping[symbol] for symbol in pauli_string])


def expectation_pauli_string(state: np.ndarray, pauli_string: str) -> float:
    operator = pauli_string_operator(pauli_string)
    return float(np.real(np.vdot(state, operator @ state)))


def expectation_hamiltonian(state: np.ndarray, terms: Sequence[tuple[float, str]]) -> float:
    return float(sum(coefficient * expectation_pauli_string(state, pauli) for coefficient, pauli in terms))


def top_measurements(probabilities: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]
