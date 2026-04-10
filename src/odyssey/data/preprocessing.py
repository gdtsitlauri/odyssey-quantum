"""Deterministic tabular preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TabularPreprocessor:
    """Simple deterministic preprocessing without heavy pipeline dependencies."""

    target_column: str
    metadata_columns: list[str] = field(default_factory=list)
    numeric_columns_: list[str] = field(default_factory=list, init=False)
    categorical_columns_: list[str] = field(default_factory=list, init=False)
    medians_: dict[str, float] = field(default_factory=dict, init=False)
    means_: dict[str, float] = field(default_factory=dict, init=False)
    stds_: dict[str, float] = field(default_factory=dict, init=False)
    categories_: dict[str, list[str]] = field(default_factory=dict, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        excluded = {self.target_column, *self.metadata_columns}
        feature_columns = [column for column in frame.columns if column not in excluded]
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []
        for column in feature_columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)
        self.numeric_columns_ = sorted(numeric_columns)
        self.categorical_columns_ = sorted(categorical_columns)
        self.feature_names_ = []
        for column in self.numeric_columns_:
            numeric_series = pd.to_numeric(frame[column], errors="coerce")
            median = float(numeric_series.median()) if not numeric_series.dropna().empty else 0.0
            filled = numeric_series.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.medians_[column] = median
            self.means_[column] = mean
            self.stds_[column] = std if std > 1e-8 else 1.0
            self.feature_names_.append(column)
        for column in self.categorical_columns_:
            values = frame[column].fillna("__missing__").astype(str)
            categories = sorted(values.unique().tolist())
            self.categories_[column] = categories
            self.feature_names_.extend([f"{column}={category}" for category in categories])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        for column in self.numeric_columns_:
            filled = pd.to_numeric(frame[column], errors="coerce").fillna(self.medians_[column]).to_numpy(dtype=np.float32)
            normalized = (filled - self.means_[column]) / self.stds_[column]
            parts.append(normalized.reshape(-1, 1))
        for column in self.categorical_columns_:
            values = frame[column].fillna("__missing__").astype(str)
            categories = self.categories_[column]
            one_hot = np.zeros((len(frame), len(categories)), dtype=np.float32)
            lookup = {category: index for index, category in enumerate(categories)}
            for row_index, value in enumerate(values):
                category_index = lookup.get(value)
                if category_index is not None:
                    one_hot[row_index, category_index] = 1.0
            parts.append(one_hot)
        if not parts:
            return np.empty((len(frame), 0), dtype=np.float32)
        return np.concatenate(parts, axis=1)

    def transform_sequences(
        self,
        frame: pd.DataFrame,
        sequence_id_column: str,
        timestamp_column: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Pack grouped rows into padded sequences for GRU-style models."""

        if sequence_id_column not in frame.columns:
            raise ValueError(f"Sequence column '{sequence_id_column}' not found for sequence packing.")
        groups = []
        labels = []
        lengths = []
        group_ids: list[str] = []
        grouped_frame = frame.sort_values([sequence_id_column, timestamp_column]).groupby(sequence_id_column)
        max_length = int(grouped_frame.size().max())
        for sequence_id, group in grouped_frame:
            group_X = self.transform(group)
            length = len(group_X)
            padded = np.zeros((max_length, group_X.shape[1]), dtype=np.float32)
            padded[:length] = group_X
            groups.append(padded)
            labels.append(float(group[self.target_column].max()))
            lengths.append(length)
            group_ids.append(str(sequence_id))
        return (
            np.stack(groups) if groups else np.empty((0, max_length, len(self.feature_names_)), dtype=np.float32),
            np.asarray(lengths, dtype=np.int64),
            np.asarray(labels, dtype=np.float32),
            group_ids,
        )

