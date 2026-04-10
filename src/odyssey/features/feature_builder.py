"""Feature augmentation helpers."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _stable_uniform(value: str, low: float, high: float) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    scaled = int(digest[:12], 16) / float(16**12 - 1)
    return low + (high - low) * scaled


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype=np.float32)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(np.float32)


def augment_public_transition_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach transparent synthetic post-quantum transition metadata to public IDS rows."""

    augmented = frame.copy()
    service = augmented["service"].astype(str) if "service" in augmented.columns else pd.Series(["unknown"] * len(augmented))
    proto = augmented["proto"].astype(str) if "proto" in augmented.columns else pd.Series(["tcp"] * len(augmented))
    state = augmented["state"].astype(str) if "state" in augmented.columns else pd.Series(["UNK"] * len(augmented))
    duration = _numeric_series(augmented, "dur", 0.0)
    rate = _numeric_series(augmented, "rate", 0.0)
    sload = _numeric_series(augmented, "sload", 0.0)
    dload = _numeric_series(augmented, "dload", 0.0)
    sloss = _numeric_series(augmented, "sloss", 0.0)
    dloss = _numeric_series(augmented, "dloss", 0.0)
    spkts = _numeric_series(augmented, "spkts", 0.0)
    dpkts = _numeric_series(augmented, "dpkts", 0.0)
    sbytes = _numeric_series(augmented, "sbytes", 0.0)
    dbytes = _numeric_series(augmented, "dbytes", 0.0)

    key_exchange = []
    mismatch = []
    archive_sensitivity = []
    observability_score = []
    noise_score = []
    certificate_age_days = []
    handshake_success_rate = []
    stealth_signal = []
    perturbation_score = []
    hndl_exposure = []
    legacy_cipher_fraction = []
    pqc_adoption_flag = []

    for idx in range(len(augmented)):
        key = f"{service.iloc[idx]}::{proto.iloc[idx]}::{state.iloc[idx]}::{idx}"
        service_name = service.iloc[idx].lower()
        proto_name = proto.iloc[idx].lower()
        state_name = state.iloc[idx].lower()
        total_packets = float(spkts.iloc[idx] + dpkts.iloc[idx] + 1.0)
        total_bytes = float(sbytes.iloc[idx] + dbytes.iloc[idx] + 1.0)
        loss_ratio = float(np.clip((sloss.iloc[idx] + dloss.iloc[idx]) / total_packets, 0.0, 1.0))
        load_pressure = float(np.clip(np.log1p(sload.iloc[idx] + dload.iloc[idx]) / 15.0, 0.0, 1.0))
        byte_asymmetry = float(np.clip(abs(sbytes.iloc[idx] - dbytes.iloc[idx]) / total_bytes, 0.0, 1.0))
        duration_score = float(np.clip(np.log1p(max(duration.iloc[idx], 0.0)) / 6.0, 0.0, 1.0))
        rate_score = float(np.clip(np.log1p(max(rate.iloc[idx], 0.0)) / 14.0, 0.0, 1.0))
        udp_factor = 1.0 if proto_name == "udp" else 0.0
        state_instability = 0.8 if state_name not in {"con", "estab", "fin"} else 0.2
        if "https" in service_name or "ssl" in service_name or "ssh" in service_name:
            kex = "ecdhe"
        elif "dns" in service_name or "ftp" in service_name or "smtp" in service_name:
            kex = "rsa_legacy"
        else:
            kex = "hybrid_pqc" if _stable_uniform(key, 0.0, 1.0) > 0.82 else "rsa_legacy"
        key_exchange.append(kex)
        legacy_fraction = 1.0 if kex == "rsa_legacy" else 0.35 if kex == "ecdhe" else 0.1
        pqc_flag = 1.0 if kex == "hybrid_pqc" else 0.0
        service_sensitivity = 0.85 if any(token in service_name for token in ["https", "ssh", "db", "storage"]) else 0.55
        archive = float(
            np.clip(
                0.45 * service_sensitivity + 0.25 * duration_score + 0.2 * (1.0 - byte_asymmetry) + 0.1 * _stable_uniform(key + "a", 0.0, 1.0),
                0.2,
                1.0,
            )
        )
        observability = float(
            np.clip(
                0.92 - 0.35 * loss_ratio - 0.18 * udp_factor - 0.15 * byte_asymmetry - 0.1 * state_instability,
                0.25,
                0.98,
            )
        )
        noise = float(np.clip(0.45 * loss_ratio + 0.35 * load_pressure + 0.2 * udp_factor, 0.02, 0.85))
        cert_age = float(np.clip(45.0 + 420.0 * legacy_fraction + 90.0 * service_sensitivity + 60.0 * _stable_uniform(key + "c", 0.0, 1.0), 15.0, 720.0))
        handshake = float(np.clip(0.97 - 0.45 * loss_ratio - 0.15 * state_instability - 0.08 * udp_factor, 0.35, 0.99))
        stealth = float(
            np.clip(
                0.45 * duration_score + 0.25 * (1.0 - rate_score) + 0.2 * (1.0 - loss_ratio) + 0.1 * (1.0 - byte_asymmetry),
                0.0,
                1.0,
            )
        )
        perturbation = float(np.clip(0.4 * noise + 0.35 * byte_asymmetry + 0.25 * load_pressure, 0.0, 1.0))
        hndl = float(np.clip(archive * (0.45 + 0.35 * legacy_fraction + 0.2 * (1.0 - observability)), 0.05, 1.0))
        mismatch_score = float(
            np.clip(
                0.5 * legacy_fraction + 0.25 * service_sensitivity + 0.15 * state_instability + 0.1 * _stable_uniform(key + "m", 0.0, 1.0) - 0.3 * pqc_flag,
                0.0,
                1.0,
            )
        )
        mismatch.append(mismatch_score)
        archive_sensitivity.append(archive)
        observability_score.append(observability)
        noise_score.append(noise)
        certificate_age_days.append(cert_age)
        handshake_success_rate.append(handshake)
        stealth_signal.append(stealth)
        perturbation_score.append(perturbation)
        hndl_exposure.append(hndl)
        legacy_cipher_fraction.append(legacy_fraction)
        pqc_adoption_flag.append(pqc_flag)

    augmented["key_exchange_family"] = key_exchange
    augmented["transition_mismatch"] = np.asarray(mismatch, dtype=np.float32)
    augmented["archive_sensitivity"] = np.asarray(archive_sensitivity, dtype=np.float32)
    augmented["observability_score"] = np.asarray(observability_score, dtype=np.float32)
    augmented["noise_score"] = np.asarray(noise_score, dtype=np.float32)
    augmented["certificate_age_days"] = np.asarray(certificate_age_days, dtype=np.float32)
    augmented["handshake_success_rate"] = np.asarray(handshake_success_rate, dtype=np.float32)
    augmented["stealth_signal"] = np.asarray(stealth_signal, dtype=np.float32)
    augmented["perturbation_score"] = np.asarray(perturbation_score, dtype=np.float32)
    augmented["hndl_exposure"] = np.asarray(hndl_exposure, dtype=np.float32)
    augmented["legacy_cipher_fraction"] = np.asarray(legacy_cipher_fraction, dtype=np.float32)
    augmented["pqc_adoption_flag"] = np.asarray(pqc_adoption_flag, dtype=np.float32)
    return augmented
