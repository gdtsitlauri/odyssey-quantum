from __future__ import annotations

from odyssey.config import load_config
from odyssey.quantum.algorithms import run_algorithm_suite
from odyssey.quantum.foundations import run_foundations_suite


def test_foundations_suite_contains_expected_demos() -> None:
    config = load_config("configs/quantum_suite.yaml")
    result = run_foundations_suite(config)
    experiments = {row["experiment"] for row in result["summary"]["state_preparation"]}
    assert "single_qubit_superposition" in experiments
    assert "bell_state" in experiments
    assert "ghz_3q" in experiments
    noise_table = result["tables"]["noise_scan"]
    assert {"depolarizing_mixture", "bell_fidelity", "entropy_bits"}.issubset(noise_table.columns)


def test_algorithm_suite_recovers_core_targets() -> None:
    config = load_config("configs/quantum_suite.yaml")
    config["quantum"]["algorithms"]["qaoa"]["gamma_steps"] = 9
    config["quantum"]["algorithms"]["qaoa"]["beta_steps"] = 9
    config["quantum"]["algorithms"]["vqe"]["maxiter"] = 35
    result = run_algorithm_suite(config)
    summary = {row["algorithm"]: row for row in result["summary"]["algorithms"] if "algorithm" in row}

    assert summary["bernstein_vazirani"]["correct"] is True
    assert summary["grover"]["correct"] is True
    assert summary["grover"]["best_success_probability"] > 0.9
    assert summary["qaoa"]["approximation_ratio"] > 0.75
    assert summary["shor_toy_reference"]["successful_factorization"] is True
    assert summary["vqe"]["absolute_error"] < 0.2
