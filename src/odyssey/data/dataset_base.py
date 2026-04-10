"""Shared dataset containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    """Raw dataset plus schema metadata before preprocessing."""

    frame: pd.DataFrame
    target_column: str = "label"
    timestamp_column: str | None = "timestamp"
    sequence_id_column: str | None = "sequence_id"
    is_synthetic: bool = False
    source_name: str = "unknown"
    original_columns: list[str] = field(default_factory=list)
    augmented_columns: list[str] = field(default_factory=list)
    excluded_feature_columns: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitArrays:
    """Preprocessed arrays for one split."""

    X: np.ndarray
    y: np.ndarray
    frame: pd.DataFrame
    feature_names: list[str]
    fragility: np.ndarray
    sequence_ids: np.ndarray
    timestamps: np.ndarray
    sequence_X: np.ndarray | None = None
    sequence_lengths: np.ndarray | None = None
    sequence_y: np.ndarray | None = None
    sequence_ids_grouped: list[str] | None = None


@dataclass
class ProcessedDataset:
    """All processed splits and preprocessing metadata."""

    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    target_column: str
    preprocessor: Any
    bundle: DatasetBundle
    metadata: dict[str, Any] = field(default_factory=dict)
