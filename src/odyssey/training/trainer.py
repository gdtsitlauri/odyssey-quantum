"""Training loops and prediction helpers."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from odyssey.data.dataset_base import ProcessedDataset, SplitArrays
from odyssey.models.baselines import make_sklearn_estimator
from odyssey.training.losses import composite_odyssey_loss


@dataclass
class TrainingResult:
    model: Any
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_score: float = float("-inf")
    train_time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OdysseyPosthocBlend:
    weight_risk: float
    temperature: float
    val_pr_auc: float
    val_brier: float
    uncertainty_gain: float = 0.0
    uncertainty_temperature_gain: float = 0.0
    uncertainty_mean: float = 0.5
    uncertainty_std: float = 0.25


@dataclass
class OdysseyTeacherEnsemble:
    member_names: list[str]
    component_names: list[str]
    component_weights: dict[str, float]
    val_pr_auc: float
    val_brier: float
    models: dict[str, Any]


class TabularTorchDataset(Dataset):
    """PyTorch dataset wrapper for tabular splits."""

    def __init__(self, split: SplitArrays) -> None:
        self.X = torch.tensor(split.X, dtype=torch.float32)
        self.y = torch.tensor(split.y, dtype=torch.float32)
        self.fragility = torch.tensor(split.fragility, dtype=torch.float32)
        self.uncertainty_targets = torch.tensor(split.uncertainty_targets, dtype=torch.float32)
        self.timestamps = torch.tensor(split.timestamps, dtype=torch.float32)
        sequence_codes = pd.Categorical(split.sequence_ids.astype(str)).codes
        self.sequence_ids = torch.tensor(sequence_codes, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.X[index],
            "y": self.y[index],
            "fragility": self.fragility[index],
            "uncertainty_targets": self.uncertainty_targets[index],
            "timestamps": self.timestamps[index],
            "sequence_ids": self.sequence_ids[index],
        }


class SequenceTorchDataset(Dataset):
    """PyTorch dataset wrapper for sequence-mode splits."""

    def __init__(self, split: SplitArrays) -> None:
        if split.sequence_X is None or split.sequence_lengths is None or split.sequence_y is None:
            raise ValueError("Sequence arrays are missing for sequence-mode training.")
        self.X = torch.tensor(split.sequence_X, dtype=torch.float32)
        self.lengths = torch.tensor(split.sequence_lengths, dtype=torch.long)
        self.y = torch.tensor(split.sequence_y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.X[index], "lengths": self.lengths[index], "y": self.y[index]}


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _create_loader(split: SplitArrays, batch_size: int, shuffle: bool, sequence_mode: bool) -> DataLoader:
    dataset = SequenceTorchDataset(split) if sequence_mode else TabularTorchDataset(split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _compute_simple_loss(outputs: dict[str, torch.Tensor], targets: torch.Tensor, positive_weight: float) -> torch.Tensor:
    pos_weight = torch.tensor(positive_weight, dtype=targets.dtype, device=targets.device)
    return F.binary_cross_entropy_with_logits(outputs["risk_logit"], targets, pos_weight=pos_weight)


def _needs_sequence_mode(model_name: str, dataset: ProcessedDataset) -> bool:
    return model_name == "gru" and dataset.train.sequence_X is not None


def _refresh_uncertainty_stats(model: Any, split: SplitArrays, device: torch.device, batch_size: int) -> None:
    if not hasattr(model, "set_uncertainty_stats"):
        return
    loader = _create_loader(split, batch_size=batch_size, shuffle=False, sequence_mode=False)
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            outputs = model(batch["x"], fragility=batch["fragility"])
            values.append(outputs["uncertainty_score"].detach().cpu())
    if values:
        stacked = torch.cat(values)
        model.set_uncertainty_stats(float(stacked.mean()), float(stacked.std(unbiased=False) + 1e-4))


def _warm_start_linear_probe(model: Any, dataset: ProcessedDataset, config: dict[str, Any]) -> None:
    if not hasattr(model, "linear_probe"):
        return
    if not bool(config["training"].get("linear_probe_warm_start", True)):
        return
    positive_weight = float(config["training"].get("positive_class_weight", 1.0))
    estimator = LogisticRegression(
        max_iter=400,
        solver="lbfgs",
        class_weight={0: 1.0, 1: max(positive_weight, 1e-3)},
    )
    estimator.fit(dataset.train.X, dataset.train.y.astype(int))
    with torch.no_grad():
        model.linear_probe.weight.copy_(torch.tensor(estimator.coef_, dtype=torch.float32))
        model.linear_probe.bias.copy_(torch.tensor(estimator.intercept_, dtype=torch.float32))


def predict_torch_model(
    model: Any,
    split: SplitArrays,
    config: dict[str, Any],
    sequence_mode: bool = False,
) -> dict[str, np.ndarray | float]:
    """Run deterministic inference and capture latency."""

    device = resolve_device(config)
    model.to(device)
    model.eval()
    batch_size = int(config["training"].get("batch_size", 64))
    loader = _create_loader(split, batch_size=batch_size, shuffle=False, sequence_mode=sequence_mode)
    probs, logits, attack_probs, attack_logits, uncertainty = [], [], [], [], []
    raw_risk_probs, raw_risk_logits = [], []
    targets = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            if sequence_mode:
                outputs = model(batch["x"], lengths=batch["lengths"])
                batch_targets = batch["y"]
            else:
                outputs = model(batch["x"], fragility=batch.get("fragility"))
                batch_targets = batch["y"]
            risk_probs = outputs["risk_prob"].detach().cpu()
            risk_logits = outputs["risk_logit"].detach().cpu()
            raw_risk_probs.append(risk_probs)
            raw_risk_logits.append(risk_logits)
            local_attack_probs = outputs.get("attack_prob")
            local_attack_logits = outputs.get("attack_logit")
            if local_attack_probs is not None:
                attack_probs.append(local_attack_probs.detach().cpu())
            if local_attack_logits is not None:
                attack_logits.append(local_attack_logits.detach().cpu())
            blend = getattr(model, "posthoc_blend", None)
            if blend is not None and local_attack_logits is not None:
                combined_logits = (
                    float(blend.weight_risk) * risk_logits
                    + (1.0 - float(blend.weight_risk)) * local_attack_logits.detach().cpu()
                )
                local_uncertainty = outputs.get("uncertainty_score")
                effective_temperature = torch.full_like(combined_logits, max(float(blend.temperature), 1e-4))
                if local_uncertainty is not None and abs(float(getattr(blend, "uncertainty_gain", 0.0))) > 1e-9:
                    centered_uncertainty = (
                        local_uncertainty.detach().cpu() - float(getattr(blend, "uncertainty_mean", 0.5))
                    ) / max(float(getattr(blend, "uncertainty_std", 0.25)), 1e-4)
                    if abs(float(getattr(blend, "uncertainty_temperature_gain", 0.0))) > 1e-9:
                        effective_temperature = effective_temperature * (
                            1.0
                            + float(blend.uncertainty_temperature_gain) * torch.relu(centered_uncertainty)
                        )
                    combined_logits = combined_logits / effective_temperature
                    combined_logits = combined_logits + float(blend.uncertainty_gain) * centered_uncertainty
                else:
                    combined_logits = combined_logits / effective_temperature
                probs.append(torch.sigmoid(combined_logits))
                logits.append(combined_logits)
            else:
                probs.append(risk_probs)
                logits.append(risk_logits)
            if "uncertainty_score" in outputs:
                uncertainty.append(outputs["uncertainty_score"].detach().cpu())
            targets.append(batch_targets.detach().cpu())
    elapsed = time.perf_counter() - start
    target_array = torch.cat(targets).numpy() if targets else np.empty(0, dtype=np.float32)
    prob_array = torch.cat(probs).numpy() if probs else np.empty(0, dtype=np.float32)
    logit_array = torch.cat(logits).numpy() if logits else np.empty(0, dtype=np.float32)
    raw_risk_prob_array = torch.cat(raw_risk_probs).numpy() if raw_risk_probs else np.empty(0, dtype=np.float32)
    raw_risk_logit_array = torch.cat(raw_risk_logits).numpy() if raw_risk_logits else np.empty(0, dtype=np.float32)
    attack_prob_array = torch.cat(attack_probs).numpy() if attack_probs else np.full_like(prob_array, 0.5)
    attack_logit_array = torch.cat(attack_logits).numpy() if attack_logits else np.zeros_like(logit_array)
    uncertainty_array = torch.cat(uncertainty).numpy() if uncertainty else np.full_like(prob_array, 0.5)
    teacher_ensemble = getattr(model, "teacher_ensemble", None)
    if teacher_ensemble is not None and not sequence_mode:
        component_values = {
            "risk_prob": raw_risk_prob_array.astype(np.float64),
            "attack_prob": attack_prob_array.astype(np.float64),
        }
        for name in teacher_ensemble.member_names:
            component_values[f"{name}_prob"] = teacher_ensemble.models[name].predict_proba(split.X)[:, 1].astype(np.float64)
        total_weight = sum(float(weight) for weight in teacher_ensemble.component_weights.values())
        if total_weight > 0.0:
            ensemble_probs = np.zeros_like(raw_risk_prob_array, dtype=np.float64)
            for name, weight in teacher_ensemble.component_weights.items():
                if name in component_values:
                    ensemble_probs += float(weight) * component_values[name]
            ensemble_probs = ensemble_probs / total_weight
        else:
            ensemble_probs = raw_risk_prob_array.astype(np.float64)
        ensemble_probs = np.clip(ensemble_probs, 1e-6, 1.0 - 1e-6)
        prob_array = ensemble_probs.astype(np.float32)
        logit_array = np.log(prob_array / (1.0 - prob_array)).astype(np.float32)
    latency_ms = (elapsed / max(1, len(target_array))) * 1000.0
    return {
        "y_true": target_array,
        "probs": prob_array,
        "logits": logit_array,
        "raw_risk_probs": raw_risk_prob_array,
        "raw_risk_logits": raw_risk_logit_array,
        "attack_probs": attack_prob_array,
        "attack_logits": attack_logit_array,
        "uncertainty": uncertainty_array,
        "latency_ms_per_sample": latency_ms,
    }


def _fit_odyssey_posthoc_blend(
    model: nn.Module,
    dataset: ProcessedDataset,
    config: dict[str, Any],
    sequence_mode: bool,
) -> OdysseyPosthocBlend | None:
    if str(config["model"]["name"]).lower() != "odyssey_risk":
        return None
    blend_cfg = config.get("training", {}).get("posthoc_blend", {})
    if blend_cfg is False or blend_cfg.get("enabled", True) is False:
        return None
    val_outputs = predict_torch_model(model, dataset.val, config, sequence_mode=sequence_mode)
    if len(np.unique(val_outputs["y_true"])) < 2:
        return None
    weight_grid = blend_cfg.get("risk_weights", [0.0, 0.15, 0.3, 0.5, 0.7, 1.0])
    temperature_grid = blend_cfg.get("temperatures", [0.85, 1.0, 1.15, 1.3])
    uncertainty_gain_grid = blend_cfg.get("uncertainty_gains", [0.0])
    uncertainty_temperature_gain_grid = blend_cfg.get("uncertainty_temperature_gains", [0.0])
    risk_logits = np.asarray(val_outputs["logits"], dtype=float)
    attack_logits = np.asarray(val_outputs["attack_logits"], dtype=float)
    if attack_logits.size == 0:
        return None
    targets = np.asarray(val_outputs["y_true"]).astype(int)
    uncertainty = np.asarray(val_outputs["uncertainty"], dtype=float)
    uncertainty_mean = float(uncertainty.mean()) if uncertainty.size else 0.5
    uncertainty_std = float(uncertainty.std()) + 1e-4 if uncertainty.size else 0.25
    standardized_uncertainty = (
        (uncertainty - uncertainty_mean) / uncertainty_std if uncertainty.size else np.zeros_like(risk_logits)
    )
    best: OdysseyPosthocBlend | None = None
    best_key = (-float("inf"), float("inf"))
    for weight_risk in weight_grid:
        for temperature in temperature_grid:
            for uncertainty_gain in uncertainty_gain_grid:
                for uncertainty_temperature_gain in uncertainty_temperature_gain_grid:
                    combined_logits = (
                        float(weight_risk) * risk_logits + (1.0 - float(weight_risk)) * attack_logits
                    )
                    effective_temperature = np.full_like(combined_logits, max(float(temperature), 1e-4))
                    if standardized_uncertainty.size:
                        effective_temperature = effective_temperature * (
                            1.0
                            + float(uncertainty_temperature_gain) * np.clip(standardized_uncertainty, 0.0, None)
                        )
                        combined_logits = combined_logits / np.clip(effective_temperature, 1e-4, None)
                        combined_logits = combined_logits + float(uncertainty_gain) * standardized_uncertainty
                    else:
                        combined_logits = combined_logits / np.clip(effective_temperature, 1e-4, None)
                    combined_probs = 1.0 / (1.0 + np.exp(-combined_logits))
                    val_pr_auc = float(average_precision_score(targets, combined_probs))
                    val_brier = float(np.mean((combined_probs - targets) ** 2))
                    key = (val_pr_auc, -val_brier)
                    if key > best_key:
                        best_key = key
                        best = OdysseyPosthocBlend(
                            weight_risk=float(weight_risk),
                            temperature=float(temperature),
                            val_pr_auc=val_pr_auc,
                            val_brier=val_brier,
                            uncertainty_gain=float(uncertainty_gain),
                            uncertainty_temperature_gain=float(uncertainty_temperature_gain),
                            uncertainty_mean=uncertainty_mean,
                            uncertainty_std=uncertainty_std,
                        )
    return best


def _fit_odyssey_teacher_ensemble(
    model: nn.Module,
    dataset: ProcessedDataset,
    config: dict[str, Any],
    sequence_mode: bool,
) -> OdysseyTeacherEnsemble | None:
    if str(config["model"]["name"]).lower() != "odyssey_risk" or sequence_mode:
        return None
    ensemble_cfg = config.get("training", {}).get("teacher_ensemble", {})
    if ensemble_cfg is False or ensemble_cfg.get("enabled", False) is False:
        return None
    member_names = list(ensemble_cfg.get("members", ["logistic_regression", "random_forest"]))
    if not member_names:
        return None

    teacher_models: dict[str, Any] = {}
    for member_name in member_names:
        if member_name not in {"logistic_regression", "random_forest"}:
            raise ValueError(f"Unsupported teacher ensemble member: {member_name}")
        result = train_sklearn_model(member_name, dataset, config, seed=int(config.get("seed", 7)))
        teacher_models[member_name] = result.model

    val_outputs = predict_torch_model(model, dataset.val, config, sequence_mode=False)
    targets = np.asarray(val_outputs["y_true"]).astype(int)
    if len(np.unique(targets)) < 2:
        return None

    feature_map: dict[str, np.ndarray] = {
        "risk_logit": np.asarray(val_outputs["raw_risk_logits"], dtype=np.float64),
        "attack_logit": np.asarray(val_outputs["attack_logits"], dtype=np.float64),
        "risk_prob": np.asarray(val_outputs["raw_risk_probs"], dtype=np.float64),
        "attack_prob": np.asarray(val_outputs["attack_probs"], dtype=np.float64),
        "uncertainty": np.asarray(val_outputs["uncertainty"], dtype=np.float64),
        "fragility": np.asarray(dataset.val.fragility, dtype=np.float64),
    }
    for member_name, estimator in teacher_models.items():
        feature_map[f"{member_name}_prob"] = estimator.predict_proba(dataset.val.X)[:, 1].astype(np.float64)

    component_names = list(
        ensemble_cfg.get(
            "components",
            ["random_forest_prob", "risk_prob", "attack_prob", "logistic_regression_prob"],
        )
    )
    component_names = [name for name in component_names if name in feature_map]
    if not component_names:
        return None
    weight_grids = ensemble_cfg.get(
        "component_weight_grid",
        {
            "random_forest_prob": [0.0, 0.4, 0.7, 1.0],
            "logistic_regression_prob": [0.0, 0.1],
            "risk_prob": [0.0, 0.15, 0.3],
            "attack_prob": [0.0, 0.1],
        },
    )
    grid_lists = [list(weight_grids.get(name, [0.0, 1.0])) for name in component_names]
    best_key = (-float("inf"), float("inf"))
    best_weights: dict[str, float] | None = None
    best_brier = float("inf")
    for candidate in np.array(np.meshgrid(*grid_lists)).T.reshape(-1, len(component_names)):
        weights = {name: float(weight) for name, weight in zip(component_names, candidate, strict=True)}
        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            continue
        blended = np.zeros_like(targets, dtype=np.float64)
        for name in component_names:
            blended += weights[name] * feature_map[name]
        blended = np.clip(blended / total_weight, 1e-6, 1.0 - 1e-6)
        val_pr_auc = float(average_precision_score(targets, blended))
        val_brier = float(np.mean((blended - targets) ** 2))
        key = (val_pr_auc, -val_brier)
        if key > best_key:
            best_key = key
            best_weights = weights
            best_brier = val_brier
    if best_weights is None:
        return None
    return OdysseyTeacherEnsemble(
        member_names=member_names,
        component_names=component_names,
        component_weights=best_weights,
        val_pr_auc=float(best_key[0]),
        val_brier=float(best_brier),
        models=teacher_models,
    )


def train_model(model: nn.Module, dataset: ProcessedDataset, config: dict[str, Any]) -> TrainingResult:
    """Train a PyTorch model with early stopping on validation PR-AUC."""

    model_name = str(config["model"]["name"]).lower()
    sequence_mode = _needs_sequence_mode(model_name, dataset)
    device = resolve_device(config)
    model.to(device)
    batch_size = int(config["training"].get("batch_size", 64))
    shuffle = not (model_name == "odyssey_risk" and float(config["training"].get("lambda_temp", 0.0)) > 0.0)
    train_loader = _create_loader(dataset.train, batch_size=batch_size, shuffle=shuffle, sequence_mode=sequence_mode)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    gradient_clip = float(config["training"].get("gradient_clip_norm", 5.0))
    patience = int(config["training"].get("patience", 4))
    epochs = int(config["training"].get("epochs", 12))
    positive_weight = float(config["training"].get("positive_class_weight", 1.0))

    if hasattr(model, "set_fragility_stats"):
        model.set_fragility_stats(float(dataset.train.fragility.mean()), float(dataset.train.fragility.std() + 1e-4))
    _warm_start_linear_probe(model, dataset, config)

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = float("-inf")
    stale_epochs = 0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            if sequence_mode:
                outputs = model(batch["x"], lengths=batch["lengths"])
                loss = _compute_simple_loss(outputs, batch["y"], positive_weight)
            else:
                outputs = model(batch["x"], fragility=batch.get("fragility"))
                if model_name == "odyssey_risk":
                    loss, _ = composite_odyssey_loss(
                        outputs,
                        batch["y"],
                        batch["sequence_ids"],
                        batch["timestamps"],
                        batch.get("uncertainty_targets"),
                        config["training"],
                    )
                else:
                    loss = _compute_simple_loss(outputs, batch["y"], positive_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            batch_size_actual = int(batch["y"].shape[0])
            running_loss += float(loss.detach().cpu()) * batch_size_actual
            seen += batch_size_actual

        if hasattr(model, "set_uncertainty_stats"):
            _refresh_uncertainty_stats(model, dataset.train, device=device, batch_size=batch_size)

        val_outputs = predict_torch_model(model, dataset.val, config, sequence_mode=sequence_mode)
        if len(np.unique(val_outputs["y_true"])) > 1:
            val_score = float(average_precision_score(val_outputs["y_true"], val_outputs["probs"]))
        else:
            val_score = 0.0
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(1, seen),
            "val_pr_auc": val_score,
        }
        history.append(epoch_record)
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    if hasattr(model, "set_uncertainty_stats"):
        _refresh_uncertainty_stats(model, dataset.train, device=device, batch_size=batch_size)
    teacher_ensemble = _fit_odyssey_teacher_ensemble(model, dataset, config, sequence_mode=sequence_mode)
    if teacher_ensemble is not None:
        model.teacher_ensemble = teacher_ensemble
        best_score = max(best_score, float(teacher_ensemble.val_pr_auc))
    posthoc_blend = _fit_odyssey_posthoc_blend(model, dataset, config, sequence_mode=sequence_mode)
    if posthoc_blend is not None:
        model.posthoc_blend = posthoc_blend
        best_score = max(best_score, float(posthoc_blend.val_pr_auc))
    train_time = time.perf_counter() - start_time
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_score=best_score,
        train_time_s=train_time,
        metadata={
            "device": str(device),
            "sequence_mode": sequence_mode,
            "posthoc_blend": None
            if posthoc_blend is None
            else {
                "weight_risk": float(posthoc_blend.weight_risk),
                "temperature": float(posthoc_blend.temperature),
                "val_pr_auc": float(posthoc_blend.val_pr_auc),
                "val_brier": float(posthoc_blend.val_brier),
                "uncertainty_gain": float(posthoc_blend.uncertainty_gain),
                "uncertainty_temperature_gain": float(posthoc_blend.uncertainty_temperature_gain),
                "uncertainty_mean": float(posthoc_blend.uncertainty_mean),
                "uncertainty_std": float(posthoc_blend.uncertainty_std),
            },
            "teacher_ensemble": None
            if teacher_ensemble is None
            else {
                "members": list(teacher_ensemble.member_names),
                "components": list(teacher_ensemble.component_names),
                "component_weights": {key: float(value) for key, value in teacher_ensemble.component_weights.items()},
                "val_pr_auc": float(teacher_ensemble.val_pr_auc),
                "val_brier": float(teacher_ensemble.val_brier),
            },
        },
    )


def train_sklearn_model(name: str, dataset: ProcessedDataset, config: dict[str, Any], seed: int) -> TrainingResult:
    """Train a scikit-learn baseline."""

    estimator = make_sklearn_estimator(name, seed)
    start = time.perf_counter()
    estimator.fit(dataset.train.X, dataset.train.y.astype(int))
    elapsed = time.perf_counter() - start
    val_probs = estimator.predict_proba(dataset.val.X)[:, 1]
    if len(np.unique(dataset.val.y)) > 1:
        best_score = float(average_precision_score(dataset.val.y, val_probs))
    else:
        best_score = 0.0
    return TrainingResult(
        model=estimator,
        best_epoch=1,
        best_val_score=best_score,
        train_time_s=elapsed,
        metadata={"device": "cpu", "sequence_mode": False},
    )

