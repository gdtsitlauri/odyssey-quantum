"""Dataset loading entrypoints and UNSW-NB15 adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    use_official_split = bool(config["data"]["public"].get("use_official_split", True))
    max_rows = config["data"]["public"].get("max_rows")

    def _prepare_frame(frame: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        prepared = frame.copy()
        prepared.columns = [column.strip() for column in prepared.columns]
        if "label" not in prepared.columns:
            if "attack_cat" not in prepared.columns:
                raise ValueError("UNSW-NB15 adapter requires either a 'label' or 'attack_cat' column.")
            prepared["label"] = (prepared["attack_cat"].astype(str).str.lower() != "normal").astype(int)
        if max_rows is not None and int(max_rows) > 0 and len(prepared) > int(max_rows):
            sampled_groups = []
            total_rows = len(prepared)
            for _, group in prepared.groupby("label"):
                sample_size = max(1, int(round(int(max_rows) * len(group) / total_rows)))
                sampled_groups.append(
                    group.sample(
                        n=min(sample_size, len(group)),
                        random_state=int(config.get("seed", 7)),
                    )
                )
            prepared = pd.concat(sampled_groups, ignore_index=True)
        if "timestamp" not in prepared.columns:
            if "stime" in prepared.columns:
                prepared["timestamp"] = pd.to_numeric(prepared["stime"], errors="coerce").ffill().fillna(0)
            else:
                prepared["timestamp"] = np.arange(len(prepared), dtype=np.int64)
        prepared["sequence_id"] = [f"{prefix}_{idx:07d}" for idx in range(len(prepared))]
        original = list(prepared.columns)
        prepared = augment_public_transition_metadata(prepared)
        augmented = [column for column in prepared.columns if column not in original]
        return prepared, original, augmented

    predefined_splits: dict[str, pd.DataFrame] | None = None
    if use_official_split and len(csv_files) >= 2 and all(path.name in {"UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"} for path in csv_files[:2]):
        train_frame, train_original_columns, train_augmented_columns = _prepare_frame(pd.read_csv(csv_files[0]), "unsw_train")
        test_frame, test_original_columns, test_augmented_columns = _prepare_frame(pd.read_csv(csv_files[1]), "unsw_test")
        frame = pd.concat([train_frame, test_frame], ignore_index=True)
        original_columns = train_original_columns
        augmented_columns = sorted(set(train_augmented_columns) | set(test_augmented_columns))
        predefined_splits = {"train": train_frame, "test": test_frame}
    else:
        frame = pd.concat((pd.read_csv(path) for path in csv_files), ignore_index=True)
        frame, original_columns, augmented_columns = _prepare_frame(frame, "unsw")

    manifest = {
        "dataset": "UNSW-NB15",
        "root_used": str(root),
        "csv_files": [str(path) for path in csv_files],
        "max_rows": int(max_rows) if max_rows is not None else None,
        "rows_loaded": int(len(frame)),
        "use_official_split": use_official_split and predefined_splits is not None,
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
        predefined_splits=predefined_splits,
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

    uncertainty_target_column = str(config.get("training", {}).get("uncertainty_target_column", "")).strip()
    if bundle.predefined_splits is not None:
        predefined_train = bundle.predefined_splits["train"].reset_index(drop=True)
        predefined_test = bundle.predefined_splits["test"].reset_index(drop=True)
        val_size = float(config["data"].get("val_size", 0.15))
        time_aware = bool(config["data"].get("time_aware_split", False)) and bundle.timestamp_column is not None
        if time_aware and bundle.timestamp_column in predefined_train.columns:
            ordered_train = predefined_train.sort_values(bundle.timestamp_column).reset_index(drop=True)
            n_val = max(1, int(round(len(ordered_train) * val_size)))
            train_frame = ordered_train.iloc[:-n_val].reset_index(drop=True)
            val_frame = ordered_train.iloc[-n_val:].reset_index(drop=True)
            if train_frame[bundle.target_column].nunique() < 2 or val_frame[bundle.target_column].nunique() < 2:
                train_frame, val_frame = train_test_split(
                    predefined_train,
                    test_size=val_size,
                    random_state=int(config.get("seed", 7)),
                    stratify=predefined_train[bundle.target_column].astype(int),
                )
                train_frame = train_frame.reset_index(drop=True)
                val_frame = val_frame.reset_index(drop=True)
        else:
            train_frame, val_frame = train_test_split(
                predefined_train,
                test_size=val_size,
                random_state=int(config.get("seed", 7)),
                stratify=predefined_train[bundle.target_column].astype(int),
            )
            train_frame = train_frame.reset_index(drop=True)
            val_frame = val_frame.reset_index(drop=True)
        split_frames = {
            "train": train_frame,
            "val": val_frame,
            "test": predefined_test,
        }
    else:
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
            for column in [
                bundle.timestamp_column,
                bundle.sequence_id_column,
                "attack_type",
                *bundle.excluded_feature_columns,
                *config["data"].get("exclude_feature_columns", []),
            ]
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
        if uncertainty_target_column and uncertainty_target_column in frame.columns:
            uncertainty_targets = pd.to_numeric(frame[uncertainty_target_column], errors="coerce").to_numpy(
                dtype=np.float32
            )
        else:
            uncertainty_targets = np.full(len(frame), np.nan, dtype=np.float32)
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
            uncertainty_targets=uncertainty_targets,
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
        "predefined_splits": bundle.predefined_splits is not None,
        "uncertainty_target_column": uncertainty_target_column or None,
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

