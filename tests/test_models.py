from __future__ import annotations

import torch

from odyssey.config import load_config
from odyssey.data.public_adapter import load_dataset, prepare_processed_dataset
from odyssey.models.quantum_head import PENNYLANE_AVAILABLE
from odyssey.models.odyssey_risk import build_model


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


