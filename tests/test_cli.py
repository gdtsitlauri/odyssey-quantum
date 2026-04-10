from __future__ import annotations

from pathlib import Path

import yaml

from odyssey.cli import build_parser, main
from odyssey.config import load_config


def test_parser_accepts_run_odyssey() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-odyssey", "--config", "configs/synthetic_small.yaml"])
    assert args.command == "run-odyssey"


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

