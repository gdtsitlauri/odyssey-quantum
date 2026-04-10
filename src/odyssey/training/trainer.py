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


class TabularTorchDataset(Dataset):
    """PyTorch dataset wrapper for tabular splits."""

    def __init__(self, split: SplitArrays) -> None:
        self.X = torch.tensor(split.X, dtype=torch.float32)
        self.y = torch.tensor(split.y, dtype=torch.float32)
        self.fragility = torch.tensor(split.fragility, dtype=torch.float32)
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
                ) / max(float(blend.temperature), 1e-4)
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
    attack_prob_array = torch.cat(attack_probs).numpy() if attack_probs else np.full_like(prob_array, 0.5)
    attack_logit_array = torch.cat(attack_logits).numpy() if attack_logits else np.zeros_like(logit_array)
    uncertainty_array = torch.cat(uncertainty).numpy() if uncertainty else np.full_like(prob_array, 0.5)
    latency_ms = (elapsed / max(1, len(target_array))) * 1000.0
    return {
        "y_true": target_array,
        "probs": prob_array,
        "logits": logit_array,
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
    risk_logits = np.asarray(val_outputs["logits"], dtype=float)
    attack_logits = np.asarray(val_outputs["attack_logits"], dtype=float)
    if attack_logits.size == 0:
        return None
    targets = np.asarray(val_outputs["y_true"]).astype(int)
    best: OdysseyPosthocBlend | None = None
    best_key = (-float("inf"), float("inf"))
    for weight_risk in weight_grid:
        for temperature in temperature_grid:
            combined_logits = (
                float(weight_risk) * risk_logits + (1.0 - float(weight_risk)) * attack_logits
            ) / max(float(temperature), 1e-4)
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
                )
    return best


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

