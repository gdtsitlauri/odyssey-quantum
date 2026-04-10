"""CSV, JSON, and Markdown reporting for experiment suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from odyssey.evaluation.ablations import export_ablation_summary
from odyssey.evaluation.metrics import evaluate_model
from odyssey.evaluation.plots import (
    plot_comparison_bars,
    plot_runtime_tradeoff,
    plot_seed_stability,
    plot_single_run_curves,
)
from odyssey.utils.io import ensure_dir, save_json, write_text


def _flatten_metric_record(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluate_model(
        record["y_true"],
        record["probs"],
        threshold=float(config["evaluation"].get("threshold", 0.5)),
        fixed_fpr_target=float(config["evaluation"].get("fixed_fpr_target", 0.05)),
        ece_bins=int(config["evaluation"].get("ece_bins", 10)),
        latency_ms_per_sample=float(record.get("latency_ms_per_sample", 0.0)),
        parameter_count_value=int(record.get("parameter_count", 0)),
        training_time_s=float(record.get("training_time_s", 0.0)),
    )
    flat = {"model_name": record["model_name"], "seed": int(record.get("seed", 0))}
    flat.update(metrics)
    flat["notes"] = record.get("notes", "")
    flat["uncertainty_mode"] = record.get("uncertainty_mode", "")
    flat["backend_used"] = record.get("backend_used", "")
    return flat


def _aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [column for column in df.columns if column not in {"model_name", "seed", "notes", "uncertainty_mode", "backend_used"}]
    grouped = df.groupby("model_name")[numeric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "model_name" if column[0] == "model_name" else f"{column[0]}_{column[1]}" for column in grouped.columns
    ]
    return grouped


def _consistency_statement(summary_df: pd.DataFrame) -> str:
    if "pr_auc_std" not in summary_df.columns or summary_df.empty:
        return "Seed stability could not be estimated."
    unstable = summary_df[summary_df["pr_auc_std"] > 0.05]
    if unstable.empty:
        return "Key PR-AUC differences were reasonably stable across the evaluated seeds."
    models = ", ".join(unstable["model_name"].tolist())
    return f"Some improvements were inconsistent across seeds, especially for: {models}."


def export_report(summary: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, destination.with_suffix(".json"))
    lines = [
        f"# {summary['experiment_name']} Report",
        "",
        f"- Dataset source: `{summary['data_source']}`",
        f"- Number of evaluated runs: `{summary['num_runs']}`",
        f"- Consistency note: {summary['consistency_note']}",
        "",
        "## Key Findings",
    ]
    for row in summary["toplines"]:
        lines.append(
            f"- `{row['model_name']}`: PR-AUC `{row.get('pr_auc_mean', float('nan')):.4f}`, "
            f"Recall `{row.get('recall_mean', float('nan')):.4f}`, "
            f"Brier `{row.get('brier_score_mean', float('nan')):.4f}`."
        )
    if summary.get("assumptions"):
        lines.extend(["", "## Research Assumptions"])
        for assumption in summary["assumptions"]:
            lines.append(f"- {assumption}")
    write_text("\n".join(lines) + "\n", destination)
    return destination


def save_suite_outputs(
    experiment_name: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    suite_type: str = "comparison",
) -> dict[str, str]:
    output_root = Path(config.get("output_dir", "outputs"))
    tables_dir = ensure_dir(output_root / "tables")
    figures_dir = ensure_dir(output_root / "figures")
    reports_dir = ensure_dir(output_root / "reports")
    include_pdf = bool(config.get("reporting", {}).get("include_pdf", True))

    flattened = [_flatten_metric_record(record, config) for record in records]
    metrics_df = pd.DataFrame(flattened).sort_values(["model_name", "seed"]).reset_index(drop=True)
    summary_df = _aggregate_summary(metrics_df)
    metrics_path = tables_dir / f"{experiment_name}_metrics.csv"
    summary_path = tables_dir / f"{experiment_name}_summary.csv"
    calibration_path = tables_dir / f"{experiment_name}_calibration.csv"
    runtime_path = tables_dir / f"{experiment_name}_runtime.csv"
    comparison_path = tables_dir / f"{experiment_name}_comparison.csv"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    summary_df[["model_name", "brier_score_mean", "ece_mean"]].to_csv(calibration_path, index=False)
    summary_df[["model_name", "latency_ms_per_sample_mean", "training_time_s_mean", "parameter_count_mean"]].to_csv(runtime_path, index=False)
    comparison_columns = [column for column in ["model_name", "accuracy_mean", "recall_mean", "f1_mean", "roc_auc_mean", "pr_auc_mean", "balanced_accuracy_mean"] if column in summary_df.columns]
    summary_df[comparison_columns].to_csv(comparison_path, index=False)

    if config.get("reporting", {}).get("save_predictions", True):
        for record in records:
            prediction_frame = pd.DataFrame(
                {
                    "y_true": record["y_true"],
                    "probability": record["probs"],
                }
            )
            prediction_frame.to_csv(
                tables_dir / f"{experiment_name}_{record['model_name']}_seed{record.get('seed', 0)}_predictions.csv",
                index=False,
            )

    for record in records:
        prefix = f"{experiment_name}_{record['model_name']}_seed{record.get('seed', 0)}"
        plot_single_run_curves(
            model_name=record["model_name"],
            y_true=np.asarray(record["y_true"]),
            probs=np.asarray(record["probs"]),
            output_dir=figures_dir,
            prefix=prefix,
            include_pdf=include_pdf,
        )

    if not summary_df.empty:
        plot_comparison_bars(summary_df, "pr_auc", figures_dir, experiment_name, include_pdf=include_pdf)
        plot_comparison_bars(summary_df, "ece", figures_dir, f"{experiment_name}_ece", include_pdf=include_pdf)
        plot_runtime_tradeoff(summary_df, figures_dir, experiment_name, include_pdf=include_pdf)
        if len(metrics_df["seed"].unique()) > 1:
            plot_seed_stability(summary_df, figures_dir, experiment_name, include_pdf=include_pdf)
        if suite_type == "ablation":
            export_ablation_summary(summary_df, tables_dir, experiment_name, include_pdf=include_pdf)

    summary_payload = {
        "experiment_name": experiment_name,
        "data_source": config["data"]["source"],
        "num_runs": len(records),
        "consistency_note": _consistency_statement(summary_df),
        "toplines": summary_df.to_dict(orient="records"),
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "summary_csv": str(summary_path),
            "comparison_csv": str(comparison_path),
            "calibration_csv": str(calibration_path),
            "runtime_csv": str(runtime_path),
        },
        "assumptions": [
            "Public-data fragility features are augmentation assumptions, not observed ground truth."
            if config["data"]["source"] != "synthetic"
            else "Synthetic runs model controlled post-quantum transition assumptions rather than real-world telemetry."
        ],
    }
    report_path = export_report(summary_payload, reports_dir / f"{experiment_name}_report.md")
    latest_path = reports_dir / "latest_metrics.json"
    save_json(summary_payload, latest_path)
    return {
        "metrics_csv": str(metrics_path),
        "summary_csv": str(summary_path),
        "report_md": str(report_path),
        "latest_json": str(latest_path),
    }


