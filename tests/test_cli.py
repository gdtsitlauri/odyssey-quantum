from __future__ import annotations

from pathlib import Path

import yaml

from odyssey.cli import build_parser, main
from odyssey.config import load_config


def test_parser_accepts_run_odyssey() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-odyssey", "--config", "configs/synthetic_small.yaml"])
    assert args.command == "run-odyssey"


def test_parser_accepts_quantum_suite() -> None:
    parser = build_parser()
    args = parser.parse_args(["quantum-suite", "--config", "configs/quantum_suite.yaml"])
    assert args.command == "quantum-suite"


def test_cli_odyssey_smoke_run() -> None:
    workspace_root = Path("outputs/pytest_smoke_workspace")
    workspace_root.mkdir(parents=True, exist_ok=True)
    config = load_config("configs/synthetic_small.yaml")
    config["experiment_name"] = "pytest_smoke"
    config["output_dir"] = str(workspace_root)
    config["data"]["synthetic"]["n_samples"] = 200
    config["data"]["synthetic"]["n_sequences"] = 50
    config["data"]["synthetic"]["window_size"] = 4
    config["training"]["epochs"] = 2
    config["training"]["patience"] = 1
    config["training"]["batch_size"] = 32
    config["search"]["enabled"] = False
    config["seeds"] = [5]
    config["model"]["quantum_enabled"] = False
    config_path = workspace_root / "smoke.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    main(["run-odyssey", "--config", str(config_path)])

    assert (workspace_root / "tables" / "pytest_smoke_metrics.csv").exists()
    assert (workspace_root / "reports" / "pytest_smoke_report.md").exists()
    assert any((workspace_root / "figures").glob("*.png"))


def test_cli_quantum_suite_smoke_run() -> None:
    workspace_root = Path("outputs/pytest_quantum_workspace")
    config = load_config("configs/quantum_suite.yaml")
    config["experiment_name"] = "pytest_quantum"
    config["output_dir"] = str(workspace_root)
    config["quantum"]["foundations"]["noise_levels"] = [0.0, 0.1]
    config["quantum"]["algorithms"]["qaoa"]["gamma_steps"] = 7
    config["quantum"]["algorithms"]["qaoa"]["beta_steps"] = 7
    config["quantum"]["algorithms"]["vqe"]["maxiter"] = 25
    config_path = workspace_root / "quantum.yaml"
    workspace_root.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    main(["quantum-suite", "--config", str(config_path)])

    assert (workspace_root / "tables" / "pytest_quantum_foundations_noise_scan.csv").exists()
    assert (workspace_root / "tables" / "pytest_quantum_algorithms_summary.csv").exists()
    assert (workspace_root / "reports" / "pytest_quantum_report.md").exists()
    assert any((workspace_root / "figures").glob("*.png"))

