"""Deterministic post-quantum transition fragility scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_fragility_scores(frame: pd.DataFrame) -> np.ndarray:
    """Compute a transparent fragility score from crypto and observability metadata."""

    default_series = lambda value: pd.Series([value] * len(frame), index=frame.index, dtype=np.float32)
    legacy_fraction = pd.to_numeric(frame.get("legacy_cipher_fraction", default_series(0.5)), errors="coerce").fillna(0.5).to_numpy(dtype=np.float32)
    mismatch = pd.to_numeric(frame.get("transition_mismatch", default_series(0.0)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    cert_age = pd.to_numeric(frame.get("certificate_age_days", default_series(90.0)), errors="coerce").fillna(90.0).to_numpy(dtype=np.float32)
    handshake_success = pd.to_numeric(frame.get("handshake_success_rate", default_series(0.85)), errors="coerce").fillna(0.85).to_numpy(dtype=np.float32)
    archive_sensitivity = pd.to_numeric(frame.get("archive_sensitivity", default_series(0.5)), errors="coerce").fillna(0.5).to_numpy(dtype=np.float32)
    observability = pd.to_numeric(frame.get("observability_score", default_series(0.8)), errors="coerce").fillna(0.8).to_numpy(dtype=np.float32)
    pqc_adoption = pd.to_numeric(frame.get("pqc_adoption_flag", default_series(0.0)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    hndl_default = pd.Series(archive_sensitivity, index=frame.index)
    hndl_exposure = pd.to_numeric(frame.get("hndl_exposure", hndl_default), errors="coerce").fillna(hndl_default).to_numpy(dtype=np.float32)

    cert_term = np.clip(cert_age / 365.0, 0.0, 2.5)
    handshake_term = 1.0 - np.clip(handshake_success, 0.0, 1.0)
    observability_term = 1.0 - np.clip(observability, 0.0, 1.0)
    raw = (
        1.8 * legacy_fraction
        + 1.4 * mismatch
        + 0.8 * cert_term
        + 1.0 * handshake_term
        + 1.1 * archive_sensitivity
        + 1.2 * hndl_exposure
        + 0.9 * observability_term
        - 1.0 * pqc_adoption
        - 1.7
    )
    return _sigmoid(raw).astype(np.float32)
