"""Experiment runners and preset registry."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

from odyssey.config import dump_config, load_config
from odyssey.data.public_adapter import load_dataset, prepare_processed_dataset
from odyssey.data.synthetic_generator import generate_synthetic_frame, save_synthetic_frame
from odyssey.evaluation.metrics import parameter_count
from odyssey.evaluation.plots import (
    plot_comparison_bars,
    plot_runtime_tradeoff,
    plot_seed_stability,
    plot_single_run_curves,
)
from odyssey.evaluation.reporting import save_suite_outputs
from odyssey.models.odyssey_risk import build_model
from odyssey.training import TemperatureScaler, predict_torch_model, train_model, train_sklearn_model
from odyssey.utils.io import ensure_dir, save_json, write_text
from odyssey.utils.seed import set_global_seed


PRESET_CONFIGS = {
    "synthetic_small_debug": "configs/synthetic_small.yaml",
    "synthetic_research_main": "configs/synthetic_research.yaml",
    "baseline_suite": "configs/baseline_suite.yaml",
    "ablation_suite": "configs/ablation_suite.yaml",
    "seed_stability_suite": "configs/seed_stability_suite.yaml",
}


def resolve_config_path(config_or_preset: str | Path) -> Path:
    candidate = Path(config_or_preset)
    if candidate.exists():
        return candidate
    preset = PRESET_CONFIGS.get(str(config_or_preset))
    if preset is None:
        raise FileNotFoundError(f"Unknown config or preset: {config_or_preset}")
    return Path(preset)


def _prepare_config(config_or_preset: str | Path) -> dict[str, Any]:
    config = load_config(resolve_config_path(config_or_preset))
    ensure_dir(Path(config.get("output_dir", "outputs")) / "logs")
    dump_config(
        config,
        Path(config.get("output_dir", "outputs")) / "logs" / f"{config['experiment_name']}_effective.yaml",
    )
    return config


def _with_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    seeded = deepcopy(config)
    seeded["seed"] = int(seed)
    seeded["data"]["synthetic"]["random_state"] = int(seed)
    return seeded


def _run_torch_model_once(config: dict[str, Any]) -> dict[str, Any]:
    set_global_seed(int(config["seed"]))
    bundle = load_dataset(config)
    processed = prepare_processed_dataset(bundle, config)
    model = build_model(config, {"input_dim": processed.train.X.shape[1]})
    result = train_model(model, processed, config)
    sequence_mode = str(config["model"]["name"]).lower() == "gru" and processed.test.sequence_X is not None
    predictions = predict_torch_model(result.model, processed.test, config, sequence_mode=sequence_mode)
    model_name = str(config["model"]["name"]).lower()
    notes = ""
    blend = result.metadata.get("posthoc_blend")
    teacher_ensemble = result.metadata.get("teacher_ensemble")
    if blend is not None:
        notes = (
            f"blend_weight_risk={float(blend['weight_risk']):.2f},"
            f"temperature={float(blend['temperature']):.2f},"
            f"uncertainty_gain={float(blend.get('uncertainty_gain', 0.0)):.2f},"
            f"uncertainty_temp_gain={float(blend.get('uncertainty_temperature_gain', 0.0)):.2f},"
            f"val_pr_auc={float(blend['val_pr_auc']):.4f}"
        )
    if teacher_ensemble is not None:
        ensemble_note = (
            f"teacher_members={','.join(teacher_ensemble['members'])},"
            f"teacher_components={','.join(teacher_ensemble['components'])},"
            f"teacher_weights={','.join(f'{key}:{value:.2f}' for key, value in teacher_ensemble['component_weights'].items())},"
            f"teacher_val_pr_auc={float(teacher_ensemble['val_pr_auc']):.4f}"
        )
        notes = f"{notes};{ensemble_note}" if notes else ensemble_note
    record = {
        "model_name": model_name,
        "seed": int(config["seed"]),
        "y_true": predictions["y_true"],
        "probs": predictions["probs"],
        "latency_ms_per_sample": predictions["latency_ms_per_sample"],
        "training_time_s": result.train_time_s,
        "parameter_count": parameter_count(result.model),
        "best_val_score": result.best_val_score,
        "notes": notes,
        "uncertainty_mode": getattr(getattr(result.model, "quantum_status", None), "mode", ""),
        "backend_used": getattr(getattr(result.model, "quantum_status", None), "backend_used", ""),
    }
    return record


def _run_mlp_calibrated_once(config: dict[str, Any]) -> dict[str, Any]:
    run_config = deepcopy(config)
    run_config["model"]["name"] = "mlp"
    set_global_seed(int(run_config["seed"]))
    bundle = load_dataset(run_config)
    processed = prepare_processed_dataset(bundle, run_config)
    model = build_model(run_config, {"input_dim": processed.train.X.shape[1]})
    result = train_model(model, processed, run_config)
    val_predictions = predict_torch_model(result.model, processed.val, run_config, sequence_mode=False)
    scaler = TemperatureScaler()
    logits = torch.tensor(val_predictions["logits"], dtype=torch.float32)
    targets = torch.tensor(val_predictions["y_true"], dtype=torch.float32)
    scaler.fit(logits, targets)
    test_predictions = predict_torch_model(result.model, processed.test, run_config, sequence_mode=False)
    calibrated_probs = (
        torch.sigmoid(scaler(torch.tensor(test_predictions["logits"], dtype=torch.float32))).detach().numpy()
    )
    return {
        "model_name": "mlp_calibrated",
        "seed": int(run_config["seed"]),
        "y_true": test_predictions["y_true"],
        "probs": calibrated_probs,
        "latency_ms_per_sample": test_predictions["latency_ms_per_sample"],
        "training_time_s": result.train_time_s,
        "parameter_count": parameter_count(result.model),
        "notes": f"temperature={float(scaler.temperature.detach().cpu()):.4f}",
        "uncertainty_mode": "",
        "backend_used": "",
    }


def _run_sklearn_once(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    set_global_seed(int(config["seed"]))
    bundle = load_dataset(config)
    processed = prepare_processed_dataset(bundle, config)
    result = train_sklearn_model(model_name, processed, config, seed=int(config["seed"]))
    start = time.perf_counter()
    probabilities = result.model.predict_proba(processed.test.X)[:, 1]
    elapsed = time.perf_counter() - start
    return {
        "model_name": model_name,
        "seed": int(config["seed"]),
        "y_true": processed.test.y,
        "probs": probabilities,
        "latency_ms_per_sample": (elapsed / max(1, len(probabilities))) * 1000.0,
        "training_time_s": result.train_time_s,
        "parameter_count": parameter_count(result.model),
        "notes": "",
        "uncertainty_mode": "",
        "backend_used": "",
    }


def _run_odyssey_stacked_ensemble_once(config: dict[str, Any]) -> dict[str, Any]:
    searched = _hyperparameter_search(config)
    set_global_seed(int(searched["seed"]))
    bundle = load_dataset(searched)
    processed = prepare_processed_dataset(bundle, searched)

    lr_result = train_sklearn_model("logistic_regression", processed, searched, seed=int(searched["seed"]))
    rf_result = train_sklearn_model("random_forest", processed, searched, seed=int(searched["seed"]))

    odyssey_config = deepcopy(searched)
    odyssey_config["model"]["name"] = "odyssey_risk"
    model = build_model(odyssey_config, {"input_dim": processed.train.X.shape[1]})
    odyssey_result = train_model(model, processed, odyssey_config)

    lr_val_probs = lr_result.model.predict_proba(processed.val.X)[:, 1]
    rf_val_probs = rf_result.model.predict_proba(processed.val.X)[:, 1]
    odyssey_val_outputs = predict_torch_model(odyssey_result.model, processed.val, odyssey_config, sequence_mode=False)
    meta_val_X = np.column_stack([lr_val_probs, rf_val_probs, odyssey_val_outputs["probs"]])
    meta_targets = processed.val.y.astype(int)
    meta_model = LogisticRegression(
        random_state=int(searched["seed"]),
        max_iter=400,
        class_weight="balanced",
        solver="lbfgs",
    )
    meta_model.fit(meta_val_X, meta_targets)

    start = time.perf_counter()
    lr_test_probs = lr_result.model.predict_proba(processed.test.X)[:, 1]
    lr_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    rf_test_probs = rf_result.model.predict_proba(processed.test.X)[:, 1]
    rf_elapsed = time.perf_counter() - start
    odyssey_test_outputs = predict_torch_model(odyssey_result.model, processed.test, odyssey_config, sequence_mode=False)

    meta_test_X = np.column_stack([lr_test_probs, rf_test_probs, odyssey_test_outputs["probs"]])
    start = time.perf_counter()
    ensemble_probs = meta_model.predict_proba(meta_test_X)[:, 1]
    meta_elapsed = time.perf_counter() - start

    meta_params = int(meta_model.coef_.size + meta_model.intercept_.size)
    total_params = (
        parameter_count(lr_result.model)
        + parameter_count(rf_result.model)
        + parameter_count(odyssey_result.model)
        + meta_params
    )
    total_training_time = lr_result.train_time_s + rf_result.train_time_s + odyssey_result.train_time_s
    total_latency = (
        ((lr_elapsed + rf_elapsed + meta_elapsed) / max(1, len(ensemble_probs))) * 1000.0
        + float(odyssey_test_outputs["latency_ms_per_sample"])
    )

    odyssey_quantum_status = getattr(odyssey_result.model, "quantum_status", None)
    note_parts = [
        "members=logistic_regression,random_forest,odyssey_risk",
        f"meta_coeffs={','.join(f'{value:.4f}' for value in meta_model.coef_[0])}",
        f"meta_bias={float(meta_model.intercept_[0]):.4f}",
    ]
    blend = odyssey_result.metadata.get("posthoc_blend")
    if blend is not None:
        note_parts.append(f"odyssey_blend_weight={float(blend['weight_risk']):.2f}")
        note_parts.append(f"odyssey_blend_temp={float(blend['temperature']):.2f}")

    return {
        "model_name": "odyssey_stacked_ensemble",
        "seed": int(searched["seed"]),
        "y_true": processed.test.y,
        "probs": ensemble_probs,
        "latency_ms_per_sample": total_latency,
        "training_time_s": total_training_time,
        "parameter_count": total_params,
        "best_val_score": float(average_precision_score(meta_targets, meta_model.predict_proba(meta_val_X)[:, 1])),
        "notes": ";".join(note_parts),
        "uncertainty_mode": getattr(odyssey_quantum_status, "mode", ""),
        "backend_used": getattr(odyssey_quantum_status, "backend_used", ""),
    }


def _apply_trial(
    config: dict[str, Any],
    learning_rate: float,
    hidden_dim: int,
    latent_dim: int,
    beta_value: float,
    gamma_value: float,
    delta_value: float,
    alpha_value: float,
    dropout_value: float,
    weight_decay: float,
    lambda_attack: float,
    disable_fragility: bool,
    quantum_enabled: bool,
    uncertainty_mode: str,
) -> dict[str, Any]:
    trial = deepcopy(config)
    trial["training"]["learning_rate"] = learning_rate
    trial["training"]["weight_decay"] = weight_decay
    trial["training"]["lambda_attack"] = lambda_attack
    trial["model"]["encoder_hidden_dim"] = hidden_dim
    trial["model"]["encoder_latent_dim"] = latent_dim
    trial["model"]["dropout"] = dropout_value
    trial["model"]["quantum_enabled"] = quantum_enabled
    trial["model"]["uncertainty_mode"] = uncertainty_mode
    trial["model"]["combiner_init"]["alpha"] = alpha_value
    trial["model"]["combiner_init"]["beta"] = beta_value
    trial["model"]["combiner_init"]["gamma"] = gamma_value
    trial["model"]["combiner_init"]["delta"] = delta_value
    trial.setdefault("features", {})
    trial["features"]["disable_fragility"] = disable_fragility
    return trial


def _hyperparameter_search(config: dict[str, Any]) -> dict[str, Any]:
    search_cfg = config.get("search", {})
    if not search_cfg.get("enabled", False):
        return config
    candidates = list(
        product(
            search_cfg.get("learning_rates", [config["training"]["learning_rate"]]),
            search_cfg.get("hidden_dims", [config["model"].get("encoder_hidden_dim", 64)]),
            search_cfg.get("latent_dims", [config["model"]["encoder_latent_dim"]]),
            search_cfg.get("beta_values", [config["model"]["combiner_init"]["beta"]]),
            search_cfg.get("gamma_values", [config["model"]["combiner_init"]["gamma"]]),
            search_cfg.get("delta_values", [config["model"]["combiner_init"]["delta"]]),
            search_cfg.get("alpha_values", [config["model"]["combiner_init"]["alpha"]]),
            search_cfg.get("dropout_values", [config["model"].get("dropout", 0.15)]),
            search_cfg.get("weight_decays", [config["training"].get("weight_decay", 1e-4)]),
            search_cfg.get("lambda_attack_values", [config["training"].get("lambda_attack", 0.0)]),
            search_cfg.get("disable_fragility_values", [config.get("features", {}).get("disable_fragility", False)]),
            search_cfg.get("quantum_enabled_values", [config["model"].get("quantum_enabled", True)]),
            search_cfg.get("uncertainty_modes", [config["model"].get("uncertainty_mode", "auto")]),
        )
    )[: int(search_cfg.get("max_trials", 6))]
    if not candidates:
        return config
    best_config = deepcopy(config)
    best_score = float("-inf")
    first_seed = int(config.get("seeds", [config["seed"]])[0])
    for (
        learning_rate,
        hidden_dim,
        latent_dim,
        beta_value,
        gamma_value,
        delta_value,
        alpha_value,
        dropout_value,
        weight_decay,
        lambda_attack,
        disable_fragility,
        quantum_enabled,
        uncertainty_mode,
    ) in candidates:
        trial = _apply_trial(
            _with_seed(config, first_seed),
            learning_rate,
            hidden_dim,
            latent_dim,
            beta_value,
            gamma_value,
            delta_value,
            alpha_value,
            dropout_value,
            weight_decay,
            lambda_attack,
            disable_fragility,
            quantum_enabled,
            uncertainty_mode,
        )
        set_global_seed(int(trial["seed"]))
        bundle = load_dataset(trial)
        processed = prepare_processed_dataset(bundle, trial)
        model = build_model(trial, {"input_dim": processed.train.X.shape[1]})
        result = train_model(model, processed, trial)
        if result.best_val_score > best_score:
            best_score = result.best_val_score
            best_config = _apply_trial(
                config,
                learning_rate,
                hidden_dim,
                latent_dim,
                beta_value,
                gamma_value,
                delta_value,
                alpha_value,
                dropout_value,
                weight_decay,
                lambda_attack,
                disable_fragility,
                quantum_enabled,
                uncertainty_mode,
            )
    return best_config


def generate_synthetic_data(config_or_preset: str | Path, output_path: str | Path | None = None) -> Path:
    config = _prepare_config(config_or_preset)
    frame = generate_synthetic_frame(config, seed=int(config["seed"]))
    destination = Path(output_path) if output_path is not None else Path("data/processed") / f"{config['experiment_name']}.csv"
    saved = save_synthetic_frame(frame, destination)
    manifest = {
        "experiment_name": config["experiment_name"],
        "rows": len(frame),
        "label_rate": float(frame["label"].mean()),
        "output_csv": str(saved),
        "synthetic": True,
    }
    save_json(manifest, Path(config.get("output_dir", "outputs")) / "logs" / f"{config['experiment_name']}_synthetic_manifest.json")
    return saved


def run_odyssey_experiment(config_or_preset: str | Path) -> dict[str, str]:
    config = _prepare_config(config_or_preset)
    searched = _hyperparameter_search(config)
    records = []
    for seed in searched.get("seeds", [searched["seed"]]):
        seed_config = _with_seed(searched, int(seed))
        seed_config["model"]["name"] = "odyssey_risk"
        records.append(_run_torch_model_once(seed_config))
    return save_suite_outputs(searched["experiment_name"], records, searched, suite_type="comparison")


def run_baseline_suite(config_or_preset: str | Path) -> dict[str, str]:
    config = _prepare_config(config_or_preset)
    models = list(config.get("baseline_models", ["logistic_regression", "random_forest", "mlp", "mlp_calibrated"]))
    records = []
    for seed in config.get("seeds", [config["seed"]]):
        seeded = _with_seed(config, int(seed))
        for model_name in models:
            if model_name in {"logistic_regression", "random_forest"}:
                records.append(_run_sklearn_once(seeded, model_name))
            elif model_name == "mlp":
                local = deepcopy(seeded)
                local["model"]["name"] = "mlp"
                records.append(_run_torch_model_once(local))
            elif model_name == "mlp_calibrated":
                records.append(_run_mlp_calibrated_once(seeded))
            elif model_name == "gru":
                if not bool(seeded["data"].get("sequence_mode", False)):
                    continue
                local = deepcopy(seeded)
                local["model"]["name"] = "gru"
                records.append(_run_torch_model_once(local))
            elif model_name == "odyssey_stacked_ensemble":
                records.append(_run_odyssey_stacked_ensemble_once(seeded))
            else:
                raise ValueError(f"Unsupported baseline entry: {model_name}")
    return save_suite_outputs(config["experiment_name"], records, config, suite_type="comparison")


def run_ablation_suite(config_or_preset: str | Path) -> dict[str, str]:
    config = _prepare_config(config_or_preset)
    records = []
    for seed in config.get("seeds", [config["seed"]]):
        seeded = _with_seed(config, int(seed))
        for ablation in config.get("ablations", ["full", "no_quantum", "no_fragility", "no_temporal", "random_uncertainty"]):
            local = deepcopy(seeded)
            local["model"]["name"] = "odyssey_risk"
            if ablation == "no_quantum":
                local["model"]["quantum_enabled"] = False
                local["model"]["uncertainty_mode"] = "zero"
            elif ablation == "no_fragility":
                local.setdefault("features", {})
                local["features"]["disable_fragility"] = True
            elif ablation == "no_temporal":
                local["training"]["lambda_temp"] = 0.0
            elif ablation == "random_uncertainty":
                local["model"]["uncertainty_mode"] = "random"
            record = _run_torch_model_once(local)
            record["model_name"] = ablation
            records.append(record)
    return save_suite_outputs(config["experiment_name"], records, config, suite_type="ablation")


def run_all_experiments(config_or_preset: str | Path) -> dict[str, str]:
    config = _prepare_config(config_or_preset)
    searched = _hyperparameter_search(config)
    records = []
    baseline_models = ["logistic_regression", "random_forest", "mlp", "mlp_calibrated"]
    for seed in searched.get("seeds", [searched["seed"]]):
        seeded = _with_seed(searched, int(seed))
        for baseline in baseline_models:
            if baseline in {"logistic_regression", "random_forest"}:
                records.append(_run_sklearn_once(seeded, baseline))
            elif baseline == "mlp":
                local = deepcopy(seeded)
                local["model"]["name"] = "mlp"
                records.append(_run_torch_model_once(local))
            else:
                records.append(_run_mlp_calibrated_once(seeded))
        local = deepcopy(seeded)
        local["model"]["name"] = "odyssey_risk"
        records.append(_run_torch_model_once(local))
    experiment_name = f"{searched['experiment_name']}_all"
    return save_suite_outputs(experiment_name, records, searched, suite_type="comparison")


def make_figures(report_path: str | Path | None = None) -> dict[str, str]:
    report_file = Path(report_path) if report_path is not None else Path("outputs/reports/latest_metrics.json")
    with report_file.open("r", encoding="utf-8") as handle:
        summary_payload = json.load(handle)
    experiment_name = summary_payload["experiment_name"]
    summary_csv = Path(summary_payload["artifacts"]["summary_csv"])
    figures_dir = Path("outputs/figures")
    summary_df = pd.read_csv(summary_csv)
    plot_comparison_bars(summary_df, "pr_auc", figures_dir, experiment_name, include_pdf=True)
    plot_runtime_tradeoff(summary_df, figures_dir, experiment_name, include_pdf=True)
    if "pr_auc_std" in summary_df.columns and summary_df["pr_auc_std"].notna().any():
        plot_seed_stability(summary_df, figures_dir, experiment_name, include_pdf=True)
    prediction_files = list(Path("outputs/tables").glob(f"{experiment_name}_*_predictions.csv"))
    for path in prediction_files:
        frame = pd.read_csv(path)
        prefix = path.stem.replace("_predictions", "")
        model_name = prefix.replace(f"{experiment_name}_", "")
        plot_single_run_curves(model_name, frame["y_true"].to_numpy(), frame["probability"].to_numpy(), figures_dir, prefix, include_pdf=True)
    return {"figures_dir": str(figures_dir)}


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = "| " + " | ".join(frame.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.to_numpy()]
    return "\n".join([headers, separator, *rows])


def export_paper_assets(source_dir: str | Path = "outputs") -> dict[str, str]:
    source_root = Path(source_dir)
    tables_dir = source_root / "tables"
    destination = ensure_dir("paper/results_tables")
    exported = []
    for csv_path in sorted(tables_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        markdown = f"# {csv_path.stem}\n\n" + _dataframe_to_markdown(frame.head(20)) + "\n"
        output_path = destination / f"{csv_path.stem}.md"
        write_text(markdown, output_path)
        exported.append(str(output_path))
    manifest_text = "# Figure Manifest Update\n\n"
    for figure in sorted((source_root / "figures").glob("*.png")):
        manifest_text += f"- `{figure.name}` -> generated experiment asset\n"
    write_text(manifest_text, "paper/results_tables/figure_manifest_generated.md")
    return {"exported_tables": "\n".join(exported)}

