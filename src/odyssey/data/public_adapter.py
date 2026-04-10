"""Dataset loading entrypoints and UNSW-NB15 adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from odyssey.data.dataset_base import DatasetBundle, ProcessedDataset, SplitArrays
from odyssey.data.preprocessing import TabularPreprocessor
from odyssey.data.splits import split_frame
from odyssey.data.synthetic_generator import generate_synthetic_frame
from odyssey.features.feature_builder import augment_public_transition_metadata
from odyssey.features.fragility_score import compute_fragility_scores


UNSW_NUMERIC_CANDIDATES = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sttl",
    "dttl",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
]
UNSW_CATEGORICAL_CANDIDATES = ["proto", "service", "state", "attack_cat"]


def _find_unsw_root(requested_root: Path) -> Path:
    if requested_root.exists():
        return requested_root
    parent = requested_root.parent
    if (parent / "UNSW_NB15_training-set.csv").exists() and (parent / "UNSW_NB15_testing-set.csv").exists():
        return parent
    raw_root = Path("data/raw")
    if (raw_root / "UNSW_NB15_training-set.csv").exists() and (raw_root / "UNSW_NB15_testing-set.csv").exists():
        return raw_root
    return requested_root


def _select_unsw_csv_files(root: Path) -> list[Path]:
    preferred = [
        root / "UNSW_NB15_training-set.csv",
        root / "UNSW_NB15_testing-set.csv",
    ]
    existing_preferred = [path for path in preferred if path.exists()]
    if existing_preferred:
        return existing_preferred
    csv_files = sorted(
        path
        for path in root.glob("*.csv")
        if "features" not in path.name.lower() and "events" not in path.name.lower()
    )
    validated: list[Path] = []
    for path in csv_files:
        try:
            sample = pd.read_csv(path, nrows=2)
        except Exception:
            continue
        sample_columns = {str(column).strip().lower() for column in sample.columns}
        if {"proto", "service", "label"}.issubset(sample_columns):
            validated.append(path)
    return validated


def _load_unsw_nb15(config: dict[str, Any]) -> DatasetBundle:
    root = _find_unsw_root(Path(config["data"]["public"]["root"]))
    if not root.exists():
        raise FileNotFoundError(
            f"UNSW-NB15 directory not found: {root}. Place CSV files under data/raw/unsw_nb15/ or data/raw/."
        )
    csv_files = _select_unsw_csv_files(root)
    if not csv_files:
        raise FileNotFoundError(
            f"No compatible UNSW-NB15 CSV files found under {root}. Expected UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv."
        )
    frame = pd.concat((pd.read_csv(path) for path in csv_files), ignore_index=True)
    frame.columns = [column.strip() for column in frame.columns]
    if "label" not in frame.columns:
        if "attack_cat" not in frame.columns:
            raise ValueError("UNSW-NB15 adapter requires either a 'label' or 'attack_cat' column.")
        frame["label"] = (frame["attack_cat"].astype(str).str.lower() != "normal").astype(int)
    max_rows = config["data"]["public"].get("max_rows")
    if max_rows is not None and int(max_rows) > 0 and len(frame) > int(max_rows):
        frame = (
            frame.groupby("label", group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=max(1, int(round(int(max_rows) * len(group) / len(frame)))),
                    random_state=int(config.get("seed", 7)),
                )
            )
            .reset_index(drop=True)
        )
    if "timestamp" not in frame.columns:
        if "stime" in frame.columns:
            frame["timestamp"] = pd.to_numeric(frame["stime"], errors="coerce").ffill().fillna(0)
        else:
            frame["timestamp"] = np.arange(len(frame), dtype=np.int64)
    frame["sequence_id"] = [f"unsw_{idx:07d}" for idx in range(len(frame))]
    original_columns = list(frame.columns)
    frame = augment_public_transition_metadata(frame)
    augmented_columns = [column for column in frame.columns if column not in original_columns]
    manifest = {
        "dataset": "UNSW-NB15",
        "root_used": str(root),
        "csv_files": [str(path) for path in csv_files],
        "max_rows": int(max_rows) if max_rows is not None else None,
        "rows_loaded": int(len(frame)),
        "original_columns": original_columns,
        "augmented_columns": augmented_columns,
        "used_numeric_candidates": [column for column in UNSW_NUMERIC_CANDIDATES if column in frame.columns],
        "used_categorical_candidates": [column for column in UNSW_CATEGORICAL_CANDIDATES if column in frame.columns],
    }
    return DatasetBundle(
        frame=frame,
        target_column="label",
        timestamp_column="timestamp",
        sequence_id_column="sequence_id",
        is_synthetic=False,
        source_name="unsw_nb15",
        original_columns=original_columns,
        augmented_columns=augmented_columns,
        excluded_feature_columns=["attack_cat"],
        manifest=manifest,
    )


def load_dataset(config: dict[str, Any]) -> DatasetBundle:
    """Load raw data according to the configured source."""

    source = config["data"]["source"]
    if source == "synthetic":
        frame = generate_synthetic_frame(config, seed=int(config.get("seed", 7)))
        return DatasetBundle(
            frame=frame,
            target_column="label",
            timestamp_column="timestamp",
            sequence_id_column="sequence_id",
            is_synthetic=True,
            source_name="synthetic_transition",
            original_columns=list(frame.columns),
            augmented_columns=[],
            excluded_feature_columns=[],
            manifest={"dataset": "synthetic", "note": "All rows are synthetic and reproducible."},
        )
    if source == "unsw_nb15":
        return _load_unsw_nb15(config)
    raise ValueError(f"Unsupported data source: {source}")


def prepare_processed_dataset(bundle: DatasetBundle, config: dict[str, Any]) -> ProcessedDataset:
    """Split, preprocess, and attach deterministic fragility features."""

    split_frames = split_frame(
        bundle.frame,
        target_column=bundle.target_column,
        sequence_id_column=bundle.sequence_id_column,
        timestamp_column=bundle.timestamp_column,
        val_size=float(config["data"].get("val_size", 0.15)),
        test_size=float(config["data"].get("test_size", 0.2)),
        time_aware=bool(config["data"].get("time_aware_split", False)) and bundle.timestamp_column is not None,
        seed=int(config.get("seed", 7)),
    )
    preprocessor = TabularPreprocessor(
        target_column=bundle.target_column,
        metadata_columns=[
            column
            for column in [bundle.timestamp_column, bundle.sequence_id_column, "attack_type", *bundle.excluded_feature_columns]
            if column is not None
        ],
    )
    preprocessor.fit(split_frames["train"])
    feature_names = preprocessor.feature_names_
    sequence_mode = bool(config["data"].get("sequence_mode", False))

    split_arrays: dict[str, SplitArrays] = {}
    for split_name, frame in split_frames.items():
        X = preprocessor.transform(frame)
        y = frame[bundle.target_column].astype(int).to_numpy()
        fragility = compute_fragility_scores(frame).astype(np.float32)
        sequence_ids = (
            frame[bundle.sequence_id_column].astype(str).to_numpy()
            if bundle.sequence_id_column and bundle.sequence_id_column in frame.columns
            else np.array([f"{split_name}_{idx}" for idx in range(len(frame))], dtype=object)
        )
        timestamps = (
            pd.to_numeric(frame[bundle.timestamp_column], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            if bundle.timestamp_column and bundle.timestamp_column in frame.columns
            else np.arange(len(frame), dtype=np.float32)
        )
        split_array = SplitArrays(
            X=X.astype(np.float32),
            y=y.astype(np.float32),
            frame=frame.reset_index(drop=True),
            feature_names=feature_names,
            fragility=fragility,
            sequence_ids=sequence_ids,
            timestamps=timestamps,
        )
        if sequence_mode:
            sequence_col = bundle.sequence_id_column or "sequence_id"
            timestamp_col = bundle.timestamp_column or "timestamp"
            sequences, lengths, labels, grouped_ids = preprocessor.transform_sequences(
                frame,
                sequence_id_column=sequence_col,
                timestamp_column=timestamp_col,
            )
            split_array.sequence_X = sequences.astype(np.float32)
            split_array.sequence_lengths = lengths.astype(np.int64)
            split_array.sequence_y = labels.astype(np.float32)
            split_array.sequence_ids_grouped = grouped_ids
        split_arrays[split_name] = split_array

    metadata = {
        "source_name": bundle.source_name,
        "is_synthetic": bundle.is_synthetic,
        "manifest": bundle.manifest,
        "sequence_mode": sequence_mode,
    }
    return ProcessedDataset(
        train=split_arrays["train"],
        val=split_arrays["val"],
        test=split_arrays["test"],
        target_column=bundle.target_column,
        preprocessor=preprocessor,
        bundle=bundle,
        metadata=metadata,
    )

