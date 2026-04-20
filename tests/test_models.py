from __future__ import annotations

import torch

from odyssey.config import load_config
from odyssey.data.public_adapter import load_dataset, prepare_processed_dataset
from odyssey.models.quantum_head import PENNYLANE_AVAILABLE
from odyssey.models.odyssey_risk import build_model
from odyssey.training.trainer import predict_torch_model, train_model


def _dataset_and_config(sequence_mode: bool = False) -> tuple[dict, any]:
    config = load_config("configs/synthetic_small.yaml")
    config["data"]["synthetic"]["n_samples"] = 240
    config["data"]["synthetic"]["n_sequences"] = 60
    config["data"]["synthetic"]["window_size"] = 4
    config["data"]["sequence_mode"] = sequence_mode
    config["search"]["enabled"] = False
    bundle = load_dataset(config)
    processed = prepare_processed_dataset(bundle, config)
    return config, processed


def test_odyssey_forward_and_quantum_fallback() -> None:
    config, processed = _dataset_and_config(sequence_mode=False)
    config["model"]["name"] = "odyssey_risk"
    config["model"]["quantum_enabled"] = True
    model = build_model(config, {"input_dim": processed.train.X.shape[1]})
    batch_x = torch.tensor(processed.train.X[:8], dtype=torch.float32)
    batch_fragility = torch.tensor(processed.train.fragility[:8], dtype=torch.float32)
    outputs = model(batch_x, fragility=batch_fragility)
    assert outputs["risk_prob"].shape == (8,)
    assert outputs["attack_prob"].shape == (8,)
    if not PENNYLANE_AVAILABLE:
        assert model.quantum_status.mode == "classical_fallback"


def test_gru_forward_sequence_mode() -> None:
    config, processed = _dataset_and_config(sequence_mode=True)
    config["model"]["name"] = "gru"
    model = build_model(config, {"input_dim": processed.train.X.shape[1]})
    batch_x = torch.tensor(processed.train.sequence_X[:4], dtype=torch.float32)
    lengths = torch.tensor(processed.train.sequence_lengths[:4], dtype=torch.long)
    outputs = model(batch_x, lengths=lengths)
    assert outputs["risk_prob"].shape == (4,)


def test_odyssey_teacher_ensemble_path() -> None:
    config, processed = _dataset_and_config(sequence_mode=False)
    config["model"]["name"] = "odyssey_risk"
    config["model"]["quantum_enabled"] = False
    config["model"]["uncertainty_mode"] = "zero"
    config["training"]["epochs"] = 2
    config["training"]["patience"] = 1
    config["training"]["batch_size"] = 32
    config["training"]["teacher_ensemble"] = {
        "enabled": True,
        "members": ["logistic_regression"],
        "components": ["risk_prob", "attack_prob", "logistic_regression_prob"],
        "component_weight_grid": {
            "risk_prob": [0.0, 0.5, 1.0],
            "attack_prob": [0.0, 0.5],
            "logistic_regression_prob": [0.0, 0.5, 1.0],
        },
    }
    model = build_model(config, {"input_dim": processed.train.X.shape[1]})
    result = train_model(model, processed, config)
    assert hasattr(result.model, "teacher_ensemble")
    predictions = predict_torch_model(result.model, processed.test, config, sequence_mode=False)
    assert predictions["probs"].shape[0] == processed.test.y.shape[0]


def test_odyssey_uncertainty_supervision_and_posthoc_gain_path() -> None:
    config, processed = _dataset_and_config(sequence_mode=False)
    config["model"]["name"] = "odyssey_risk"
    config["training"]["epochs"] = 2
    config["training"]["patience"] = 1
    config["training"]["batch_size"] = 32
    config["training"]["uncertainty_target_column"] = "uncertainty_hint"
    config["training"]["lambda_uncertainty"] = 0.25
    config["training"]["lambda_uncertainty_corr"] = 0.1
    config["training"]["posthoc_blend"] = {
        "enabled": True,
        "risk_weights": [0.0, 0.5, 1.0],
        "temperatures": [0.8, 1.0],
        "uncertainty_gains": [0.0],
        "uncertainty_temperature_gains": [0.0, 0.5],
    }
    model = build_model(config, {"input_dim": processed.train.X.shape[1]})
    result = train_model(model, processed, config)
    assert hasattr(result.model, "posthoc_blend")
    assert result.model.posthoc_blend.uncertainty_std > 0.0
    predictions = predict_torch_model(result.model, processed.test, config, sequence_mode=False)
    assert predictions["uncertainty"].shape[0] == processed.test.y.shape[0]


