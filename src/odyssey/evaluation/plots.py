"""Plotting utilities for publication-oriented outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn import metrics as skm


def _save_figure(fig: plt.Figure, output_prefix: Path, include_pdf: bool = True) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    if include_pdf:
        fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_single_run_curves(
    model_name: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    output_dir: str | Path,
    prefix: str,
    include_pdf: bool = True,
) -> None:
    output_base = Path(output_dir)
    preds = (probs >= 0.5).astype(int)

    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = skm.roc_curve(y_true, probs)
        precision, recall, _ = skm.precision_recall_curve(y_true, probs)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=model_name, linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"ROC: {model_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        _save_figure(fig, output_base / f"{prefix}_roc", include_pdf=include_pdf)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, label=model_name, linewidth=2)
        ax.set_title(f"Precision-Recall: {model_name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left")
        _save_figure(fig, output_base / f"{prefix}_pr", include_pdf=include_pdf)

        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"Calibration: {model_name}")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction Positive")
        _save_figure(fig, output_base / f"{prefix}_calibration", include_pdf=include_pdf)

    conf = skm.confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(conf, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=["Benign", "Attack"])
    ax.set_yticks([0, 1], labels=["Benign", "Attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(conf[i, j]), ha="center", va="center", color="black")
    _save_figure(fig, output_base / f"{prefix}_confusion", include_pdf=include_pdf)


def plot_comparison_bars(summary_df: pd.DataFrame, metric: str, output_dir: str | Path, prefix: str, include_pdf: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary_df["model_name"], summary_df[f"{metric}_mean"], yerr=summary_df.get(f"{metric}_std"), capsize=4)
    ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=25)
    _save_figure(fig, Path(output_dir) / f"{prefix}_{metric}_bar", include_pdf=include_pdf)


def plot_seed_stability(summary_df: pd.DataFrame, output_dir: str | Path, prefix: str, include_pdf: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(summary_df["model_name"], summary_df["pr_auc_mean"], yerr=summary_df["pr_auc_std"], fmt="o", capsize=4)
    ax.set_title("Seed Stability (PR-AUC)")
    ax.set_ylabel("PR-AUC")
    ax.tick_params(axis="x", rotation=25)
    _save_figure(fig, Path(output_dir) / f"{prefix}_seed_stability", include_pdf=include_pdf)


def plot_runtime_tradeoff(summary_df: pd.DataFrame, output_dir: str | Path, prefix: str, include_pdf: bool = True) -> None:
    finite_df = summary_df[
        np.isfinite(summary_df["latency_ms_per_sample_mean"].to_numpy())
        & np.isfinite(summary_df["pr_auc_mean"].to_numpy())
    ].copy()
    if finite_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(finite_df["latency_ms_per_sample_mean"], finite_df["pr_auc_mean"], s=80)
    for _, row in finite_df.iterrows():
        ax.text(row["latency_ms_per_sample_mean"], row["pr_auc_mean"], row["model_name"], fontsize=8)
    ax.set_title("Runtime vs PR-AUC")
    ax.set_xlabel("Latency per Sample (ms)")
    ax.set_ylabel("PR-AUC")
    _save_figure(fig, Path(output_dir) / f"{prefix}_runtime_tradeoff", include_pdf=include_pdf)
