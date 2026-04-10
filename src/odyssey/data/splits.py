"""Reproducible split utilities with group-aware handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _has_both_classes(frame: pd.DataFrame, target_column: str) -> bool:
    return frame[target_column].nunique() >= 2


def _group_index(
    frame: pd.DataFrame,
    target_column: str,
    sequence_id_column: str | None,
    timestamp_column: str | None,
) -> pd.DataFrame:
    if sequence_id_column and sequence_id_column in frame.columns:
        grouped = frame.groupby(sequence_id_column)
        return pd.DataFrame(
            {
                sequence_id_column: list(grouped.groups.keys()),
                "group_label": grouped[target_column].max().to_numpy(),
                "group_time": grouped[timestamp_column].min().to_numpy()
                if timestamp_column and timestamp_column in frame.columns
                else np.arange(grouped.ngroups),
            }
        )
    fallback_id = "__row_group_id__"
    temp = frame.copy()
    temp[fallback_id] = np.arange(len(temp))
    return pd.DataFrame(
        {
            fallback_id: temp[fallback_id].to_numpy(),
            "group_label": temp[target_column].astype(int).to_numpy(),
            "group_time": temp[timestamp_column].to_numpy() if timestamp_column and timestamp_column in temp.columns else np.arange(len(temp)),
        }
    )


def split_frame(
    frame: pd.DataFrame,
    target_column: str,
    sequence_id_column: str | None,
    timestamp_column: str | None,
    val_size: float,
    test_size: float,
    time_aware: bool,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Split rows while keeping sequences intact when available."""

    index_frame = _group_index(frame, target_column, sequence_id_column, timestamp_column)
    group_col = sequence_id_column if sequence_id_column and sequence_id_column in frame.columns else index_frame.columns[0]

    def _materialize(train_groups: list, val_groups: list, test_groups: list) -> dict[str, pd.DataFrame]:
        if sequence_id_column and sequence_id_column in frame.columns:
            return {
                "train": frame[frame[sequence_id_column].isin(train_groups)].sort_values([sequence_id_column, timestamp_column or sequence_id_column]).reset_index(drop=True),
                "val": frame[frame[sequence_id_column].isin(val_groups)].sort_values([sequence_id_column, timestamp_column or sequence_id_column]).reset_index(drop=True),
                "test": frame[frame[sequence_id_column].isin(test_groups)].sort_values([sequence_id_column, timestamp_column or sequence_id_column]).reset_index(drop=True),
            }

        temp = frame.copy()
        temp["__row_group_id__"] = np.arange(len(temp))
        splits = {
            "train": temp[temp["__row_group_id__"].isin(train_groups)].drop(columns="__row_group_id__"),
            "val": temp[temp["__row_group_id__"].isin(val_groups)].drop(columns="__row_group_id__"),
            "test": temp[temp["__row_group_id__"].isin(test_groups)].drop(columns="__row_group_id__"),
        }
        return {key: value.reset_index(drop=True) for key, value in splits.items()}

    if time_aware:
        ordered = index_frame.sort_values("group_time").reset_index(drop=True)
        n_groups = len(ordered)
        n_test = max(1, int(round(n_groups * test_size)))
        n_val = max(1, int(round(n_groups * val_size)))
        n_train = max(1, n_groups - n_test - n_val)
        train_groups = ordered.iloc[:n_train][group_col].tolist()
        val_groups = ordered.iloc[n_train : n_train + n_val][group_col].tolist()
        test_groups = ordered.iloc[n_train + n_val :][group_col].tolist()
        candidate = _materialize(train_groups, val_groups, test_groups)
        if all(_has_both_classes(split, target_column) for split in candidate.values()):
            return candidate

    else:
        pass

    train_val_groups, test_groups = train_test_split(
        index_frame[group_col],
        test_size=test_size,
        random_state=seed,
        stratify=index_frame["group_label"] if index_frame["group_label"].nunique() > 1 else None,
    )
    remaining = index_frame[index_frame[group_col].isin(train_val_groups)]
    adjusted_val_size = val_size / max(1e-8, 1.0 - test_size)
    train_groups, val_groups = train_test_split(
        remaining[group_col],
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=remaining["group_label"] if remaining["group_label"].nunique() > 1 else None,
    )
    return _materialize(train_groups.tolist(), val_groups.tolist(), test_groups.tolist())
