"""Config-driven workflows for Odyssey's quantum track."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from odyssey.config import load_config
from odyssey.quantum.algorithms import run_algorithm_suite
from odyssey.quantum.foundations import run_foundations_suite
from odyssey.quantum.reporting import (
    make_algorithm_figure,
    make_foundations_figure,
    render_quantum_markdown,
    save_json_summary,
    save_markdown_report,
    save_tables,
)
from odyssey.utils.io import ensure_dir


def _resolve_output_root(config: dict[str, Any]) -> Path:
    return ensure_dir(config.get("output_dir", "results/quantum_suite"))


def _result_prefix(config: dict[str, Any], suffix: str) -> str:
    return f"{config.get('experiment_name', 'quantum')}_{suffix}"


def _prepare_dirs(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "tables": ensure_dir(root / "tables"),
        "reports": ensure_dir(root / "reports"),
        "figures": ensure_dir(root / "figures"),
    }


def _load_workflow_config(config_path: str | Path) -> dict[str, Any]:
    return load_config(config_path)


def run_quantum_foundations_workflow(config_path: str | Path) -> Path:
    config = _load_workflow_config(config_path)
    root = _resolve_output_root(config)
    dirs = _prepare_dirs(root)
    payload = run_foundations_suite(config)
    prefix = _result_prefix(config, "foundations")
    table_paths = save_tables(payload["tables"], dirs["tables"], prefix)
    save_json_summary(payload["summary"], dirs["reports"], f"{prefix}_summary.json")
    make_foundations_figure(payload["tables"]["noise_scan"], dirs["figures"], f"{prefix}_noise.png")
    report_text = render_quantum_markdown("Odyssey Quantum Foundations", payload["summary"], table_paths)
    report_path = save_markdown_report(report_text, dirs["reports"], f"{prefix}_report.md")
    return Path(report_path)


def run_quantum_algorithms_workflow(config_path: str | Path) -> Path:
    config = _load_workflow_config(config_path)
    root = _resolve_output_root(config)
    dirs = _prepare_dirs(root)
    payload = run_algorithm_suite(config)
    prefix = _result_prefix(config, "algorithms")
    table_paths = save_tables(payload["tables"], dirs["tables"], prefix)
    save_json_summary(payload["summary"], dirs["reports"], f"{prefix}_summary.json")
    make_algorithm_figure(payload["tables"]["summary"], dirs["figures"], f"{prefix}_snapshot.png")
    report_text = render_quantum_markdown("Odyssey Quantum Algorithms", payload["summary"], table_paths)
    report_path = save_markdown_report(report_text, dirs["reports"], f"{prefix}_report.md")
    return Path(report_path)


def run_quantum_suite_workflow(config_path: str | Path) -> Path:
    config = _load_workflow_config(config_path)
    root = _resolve_output_root(config)
    dirs = _prepare_dirs(root)

    foundations_payload = run_foundations_suite(config)
    algorithms_payload = run_algorithm_suite(config)
    combined_summary = {
        "suite": "quantum_suite",
        "backend_status": foundations_payload["summary"]["backend_status"],
        "state_preparation": foundations_payload["summary"]["state_preparation"],
        "noise_scan": foundations_payload["summary"]["noise_scan"],
        "algorithms": algorithms_payload["summary"]["algorithms"],
    }
    table_paths = {}
    prefix = config.get("experiment_name", "quantum_suite")
    table_paths.update(save_tables(foundations_payload["tables"], dirs["tables"], f"{prefix}_foundations"))
    table_paths.update(save_tables(algorithms_payload["tables"], dirs["tables"], f"{prefix}_algorithms"))
    save_json_summary(combined_summary, dirs["reports"], f"{prefix}_summary.json")
    make_foundations_figure(
        foundations_payload["tables"]["noise_scan"],
        dirs["figures"],
        f"{prefix}_foundations_noise.png",
    )
    make_algorithm_figure(
        algorithms_payload["tables"]["summary"],
        dirs["figures"],
        f"{prefix}_algorithms_snapshot.png",
    )
    report_text = render_quantum_markdown("Odyssey Quantum Track", combined_summary, table_paths)
    report_path = save_markdown_report(report_text, dirs["reports"], f"{prefix}_report.md")
    return Path(report_path)
