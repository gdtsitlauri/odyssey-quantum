from __future__ import annotations

import numpy as np

from odyssey.config import load_config
from odyssey.data.public_adapter import load_dataset, prepare_processed_dataset
from odyssey.data.synthetic_generator import generate_synthetic_frame


def _small_config() -> dict:
    config = load_config("configs/synthetic_small.yaml")
    config["data"]["synthetic"]["n_samples"] = 240
    config["data"]["synthetic"]["n_sequences"] = 60
    config["data"]["synthetic"]["window_size"] = 4
    config["data"]["sequence_mode"] = True
    config["search"]["enabled"] = False
    return config


def test_synthetic_generation_is_deterministic() -> None:
    config = _small_config()
    frame_a = generate_synthetic_frame(config, seed=17)
    frame_b = generate_synthetic_frame(config, seed=17)
    assert frame_a.equals(frame_b)
    assert set(frame_a["attack_type"].unique()) >= {"benign"}


def test_processed_dataset_shapes_and_fragility_bounds() -> None:
    config = _small_config()
    bundle = load_dataset(config)
    processed = prepare_processed_dataset(bundle, config)
    assert processed.train.X.shape[1] == processed.val.X.shape[1] == processed.test.X.shape[1]
    assert processed.train.X.shape[0] > 0
    assert processed.train.sequence_X is not None
    assert processed.train.sequence_lengths is not None
    assert np.all((processed.train.fragility >= 0.0) & (processed.train.fragility <= 1.0))


