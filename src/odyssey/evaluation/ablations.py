"""Ablation reporting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from odyssey.evaluation.plots import plot_comparison_bars


def export_ablation_summary(summary_df: pd.DataFrame, output_dir: str | Path, prefix: str, include_pdf: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_table = summary_df.copy()
    path = output_dir / f"{prefix}_ablation_summary.csv"
    ablation_table.to_csv(path, index=False)
    plot_comparison_bars(ablation_table, "pr_auc", output_dir, f"{prefix}_ablation", include_pdf=include_pdf)
    return path


