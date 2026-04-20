"""Reporting helpers for Odyssey's quantum track."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from odyssey.utils.io import ensure_dir, save_json, write_text


def save_tables(tables: dict[str, pd.DataFrame], destination_dir: str | Path, prefix: str) -> dict[str, str]:
    destination = ensure_dir(destination_dir)
    table_paths: dict[str, str] = {}
    for name, table in tables.items():
        path = destination / f"{prefix}_{name}.csv"
        table.to_csv(path, index=False)
        table_paths[name] = str(path)
    return table_paths


def save_json_summary(summary: dict[str, Any], destination_dir: str | Path, filename: str) -> str:
    path = Path(destination_dir) / filename
    save_json(summary, path)
    return str(path)


def render_quantum_markdown(title: str, summary: dict[str, Any], table_paths: dict[str, str]) -> str:
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Suite: `{summary.get('suite', 'quantum')}`",
    ]
    backend_status = summary.get("backend_status")
    if isinstance(backend_status, dict):
        lines.append(f"- Preferred backend: `{backend_status.get('preferred_backend', 'unknown')}`")
        lines.append(
            "- Backend availability: "
            f"PennyLane={backend_status.get('pennylane_available')} | "
            f"Qiskit={backend_status.get('qiskit_available')} | "
            f"Aer={backend_status.get('qiskit_aer_available')}"
        )
    lines.append("")
    if "state_preparation" in summary:
        lines.extend(["## Foundations", ""])
        for row in summary["state_preparation"]:
            lines.append(
                f"- `{row['experiment']}` top state `{row['top_state']}` "
                f"with probability {row['top_probability']:.4f}"
            )
        lines.append("")
    if "algorithms" in summary:
        lines.extend(["## Algorithms", ""])
        for row in summary["algorithms"]:
            algorithm = row.get("algorithm", "unknown")
            if algorithm == "bernstein_vazirani":
                lines.append(
                    f"- `bernstein_vazirani` recovered `{row['recovered_string']}` "
                    f"from hidden string `{row['hidden_string']}` with success {row['success_probability']:.4f}"
                )
            elif algorithm == "grover":
                lines.append(
                    f"- `grover` best measurement `{row['best_measurement']}` "
                    f"for marked state `{row['marked_state']}` with success {row['best_success_probability']:.4f}"
                )
            elif algorithm == "vqe":
                lines.append(
                    f"- `vqe` estimated ground energy {row['ground_energy_estimate']:.6f} "
                    f"(error {row['absolute_error']:.6f})"
                )
            elif algorithm == "qaoa":
                lines.append(
                    f"- `qaoa` reached approximation ratio {row['approximation_ratio']:.4f} "
                    f"on `{row['graph']}`"
                )
            elif algorithm == "shor_toy_reference":
                lines.append(
                    f"- `shor_toy_reference` factored {row['composite']} "
                    f"into {row['factor_left']} x {row['factor_right']} with order {row['order']}"
                )
            elif algorithm == "deutsch_jozsa":
                lines.append(
                    f"- `deutsch_jozsa` classified `{row['oracle']}` correctly={row['correct']}"
                )
        lines.append("")
    lines.extend(["## Artifacts", ""])
    for name, path in table_paths.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def save_markdown_report(text: str, destination_dir: str | Path, filename: str) -> str:
    path = Path(destination_dir) / filename
    write_text(text, path)
    return str(path)


def make_foundations_figure(noise_table: pd.DataFrame, destination_dir: str | Path, filename: str) -> str:
    ensure_dir(destination_dir)
    path = Path(destination_dir) / filename
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(noise_table["depolarizing_mixture"], noise_table["bell_fidelity"], marker="o", label="Bell fidelity")
    ax.plot(noise_table["depolarizing_mixture"], noise_table["entropy_bits"], marker="s", label="Entropy")
    ax.set_xlabel("Depolarizing mixture")
    ax.set_ylabel("Score")
    ax.set_title("Odyssey Quantum Foundations: Bell Noise Scan")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def make_algorithm_figure(summary_table: pd.DataFrame, destination_dir: str | Path, filename: str) -> str:
    ensure_dir(destination_dir)
    path = Path(destination_dir) / filename
    scores = []
    labels = []
    for _, row in summary_table.iterrows():
        algorithm = row.get("algorithm", "")
        if algorithm == "grover":
            labels.append("grover")
            scores.append(float(row["best_success_probability"]))
        elif algorithm == "vqe":
            labels.append("vqe_err")
            scores.append(float(max(0.0, 1.0 - row["absolute_error"])))
        elif algorithm == "qaoa":
            labels.append("qaoa")
            scores.append(float(row["approximation_ratio"]))
        elif algorithm == "bernstein_vazirani":
            labels.append("bv")
            scores.append(float(row["success_probability"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, scores, color=["#0d3b66", "#f4a261", "#2a9d8f", "#7f5539"][: len(labels)])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Normalized score")
    ax.set_title("Odyssey Quantum Algorithms Snapshot")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)
