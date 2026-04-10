"""Synthetic benchmark generator for Odyssey transition scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CRYPTO_FAMILIES = ["rsa_legacy", "ecdhe", "hybrid_pqc", "kyber_tls"]
SERVICES = ["http", "https", "dns", "ssh", "smtp", "db", "storage"]
PROTOCOLS = ["tcp", "udp"]


@dataclass
class SyntheticConfig:
    n_samples: int
    n_sequences: int
    window_size: int
    attack_rate: float
    stealth_fraction: float
    noisy_fraction: float
    fragility_fraction: float
    random_state: int


def _sample_scenario(
    rng: np.random.Generator,
    attack_rate: float,
    stealth_fraction: float,
    noisy_fraction: float,
    fragility_fraction: float,
) -> tuple[str, int]:
    is_attack = int(rng.random() < attack_rate)
    if not is_attack:
        if rng.random() < fragility_fraction * 0.45:
            return "benign_fragile", 0
        return "benign", 0
    draw = rng.random()
    if draw < stealth_fraction:
        return "stealth_attack", 1
    if draw < stealth_fraction + noisy_fraction:
        return "noisy_attack", 1
    if draw < stealth_fraction + noisy_fraction + fragility_fraction:
        return "fragility_attack", 1
    return "conventional_attack", 1


def _bounded(value: np.ndarray | float, lower: float = 0.0, upper: float = 1.0) -> np.ndarray | float:
    return np.clip(value, lower, upper)


def generate_synthetic_frame(config: dict[str, Any], seed: int | None = None) -> pd.DataFrame:
    """Generate the fully labeled synthetic benchmark frame."""

    synth_cfg = config["data"]["synthetic"]
    cfg = SyntheticConfig(
        n_samples=int(synth_cfg["n_samples"]),
        n_sequences=int(synth_cfg["n_sequences"]),
        window_size=int(synth_cfg["window_size"]),
        attack_rate=float(synth_cfg["attack_rate"]),
        stealth_fraction=float(synth_cfg["stealth_fraction"]),
        noisy_fraction=float(synth_cfg["noisy_fraction"]),
        fragility_fraction=float(synth_cfg["fragility_fraction"]),
        random_state=int(seed if seed is not None else synth_cfg.get("random_state", config.get("seed", 7))),
    )
    rng = np.random.default_rng(cfg.random_state)
    n_sequences = max(1, cfg.n_sequences)
    windows = max(1, cfg.window_size)
    n_rows = max(cfg.n_samples, n_sequences * windows)
    rows: list[dict[str, Any]] = []
    global_index = 0

    for seq_idx in range(n_sequences):
        scenario, label = _sample_scenario(
            rng,
            cfg.attack_rate,
            cfg.stealth_fraction,
            cfg.noisy_fraction,
            cfg.fragility_fraction,
        )
        service = rng.choice(SERVICES)
        protocol = rng.choice(PROTOCOLS, p=[0.84, 0.16])
        archival_value = 0.2 + 0.6 * rng.random()
        base_timestamp = 1_700_000_000 + seq_idx * (windows + int(rng.integers(1, 4)))
        key_exchange = rng.choice(
            CRYPTO_FAMILIES,
            p=[0.45, 0.3, 0.18, 0.07] if scenario in {"benign", "conventional_attack"} else [0.55, 0.22, 0.18, 0.05],
        )
        for window_idx in range(windows):
            if global_index >= n_rows:
                break
            progress = window_idx / max(1, windows - 1)
            observability = rng.uniform(0.7, 0.98)
            noise = rng.uniform(0.02, 0.18)
            packet_count = rng.normal(40, 8)
            failed_auth = rng.uniform(0.0, 0.06)
            port_entropy = rng.normal(0.28, 0.07)
            cpu_load = rng.uniform(0.2, 0.55)
            mem_pressure = rng.uniform(0.15, 0.45)
            handshake_success = rng.uniform(0.87, 0.99)
            cert_age = rng.uniform(20, 180)
            mismatch = 0.0
            hndl_risk = archival_value * rng.uniform(0.2, 0.7)
            stealth_signal = 0.0
            perturbation = 0.0

            if scenario == "conventional_attack":
                packet_count = rng.normal(88, 18)
                failed_auth = rng.uniform(0.12, 0.45)
                port_entropy = rng.normal(0.66, 0.12)
                cpu_load = rng.uniform(0.45, 0.95)
                mem_pressure = rng.uniform(0.35, 0.82)
                handshake_success = rng.uniform(0.48, 0.84)
                mismatch = rng.uniform(0.18, 0.62)
                hndl_risk = archival_value * rng.uniform(0.45, 0.92)
            elif scenario == "stealth_attack":
                packet_count = rng.normal(36 + progress * 14, 6)
                failed_auth = rng.uniform(0.03, 0.13)
                port_entropy = rng.normal(0.4 + progress * 0.1, 0.05)
                cpu_load = rng.uniform(0.28, 0.62)
                mem_pressure = rng.uniform(0.22, 0.55)
                observability = rng.uniform(0.42, 0.82)
                noise = rng.uniform(0.07, 0.24)
                handshake_success = rng.uniform(0.62, 0.9)
                mismatch = rng.uniform(0.25, 0.55)
                stealth_signal = 0.55 + 0.35 * progress
                hndl_risk = archival_value * rng.uniform(0.55, 0.95)
            elif scenario == "noisy_attack":
                packet_count = rng.normal(68, 14)
                failed_auth = rng.uniform(0.08, 0.24)
                port_entropy = rng.normal(0.57, 0.11)
                cpu_load = rng.uniform(0.38, 0.78)
                mem_pressure = rng.uniform(0.3, 0.72)
                observability = rng.uniform(0.3, 0.72)
                noise = rng.uniform(0.22, 0.62)
                handshake_success = rng.uniform(0.54, 0.88)
                mismatch = rng.uniform(0.2, 0.58)
                perturbation = rng.uniform(0.55, 0.95)
                hndl_risk = archival_value * rng.uniform(0.4, 0.9)
            elif scenario == "fragility_attack":
                packet_count = rng.normal(54, 11)
                failed_auth = rng.uniform(0.05, 0.18)
                port_entropy = rng.normal(0.49, 0.08)
                cpu_load = rng.uniform(0.34, 0.75)
                mem_pressure = rng.uniform(0.25, 0.68)
                observability = rng.uniform(0.5, 0.88)
                noise = rng.uniform(0.08, 0.22)
                handshake_success = rng.uniform(0.45, 0.78)
                mismatch = rng.uniform(0.48, 0.96)
                cert_age = rng.uniform(150, 720)
                hndl_risk = archival_value * rng.uniform(0.65, 1.0)
                key_exchange = rng.choice(["rsa_legacy", "ecdhe", "hybrid_pqc"], p=[0.58, 0.27, 0.15])
            elif scenario == "benign_fragile":
                packet_count = rng.normal(38, 7)
                failed_auth = rng.uniform(0.0, 0.05)
                port_entropy = rng.normal(0.3, 0.06)
                cpu_load = rng.uniform(0.22, 0.5)
                mem_pressure = rng.uniform(0.16, 0.45)
                observability = rng.uniform(0.52, 0.92)
                noise = rng.uniform(0.04, 0.16)
                handshake_success = rng.uniform(0.68, 0.93)
                mismatch = rng.uniform(0.35, 0.82)
                cert_age = rng.uniform(120, 540)
                hndl_risk = archival_value * rng.uniform(0.5, 0.95)

            bytes_in = max(24.0, packet_count * rng.uniform(18, 44))
            bytes_out = max(22.0, packet_count * rng.uniform(16, 42))
            duration = max(1.0, rng.normal(18 + progress * 3, 6))
            jitter = abs(rng.normal(noise * 12, 1.2))
            packet_loss = _bounded(noise * rng.uniform(0.4, 1.1), 0.0, 1.0)
            legacy_fraction = 1.0 if key_exchange == "rsa_legacy" else 0.35 if key_exchange == "ecdhe" else 0.1
            pqc_adoption = 1.0 if key_exchange in {"hybrid_pqc", "kyber_tls"} else 0.0
            uncertainty_hint = _bounded((1.0 - observability) * 0.55 + noise * 0.45, 0.0, 1.0)
            rows.append(
                {
                    "timestamp": base_timestamp + window_idx,
                    "sequence_id": f"seq_{seq_idx:05d}",
                    "window_index": window_idx,
                    "duration": duration,
                    "packet_count": max(1.0, packet_count),
                    "bytes_in": bytes_in,
                    "bytes_out": bytes_out,
                    "flow_rate": (bytes_in + bytes_out) / max(duration, 1.0),
                    "failed_auth_rate": failed_auth,
                    "port_entropy": max(0.01, port_entropy),
                    "cpu_load": cpu_load,
                    "mem_pressure": mem_pressure,
                    "observability_score": observability,
                    "noise_score": noise,
                    "handshake_success_rate": handshake_success,
                    "certificate_age_days": cert_age,
                    "archive_sensitivity": archival_value,
                    "hndl_exposure": hndl_risk,
                    "jitter": jitter,
                    "packet_loss_rate": packet_loss,
                    "legacy_cipher_fraction": legacy_fraction,
                    "pqc_adoption_flag": pqc_adoption,
                    "transition_mismatch": mismatch,
                    "stealth_signal": stealth_signal,
                    "perturbation_score": perturbation,
                    "uncertainty_hint": uncertainty_hint,
                    "key_exchange_family": key_exchange,
                    "service": service,
                    "protocol": protocol,
                    "attack_type": scenario,
                    "label": label,
                }
            )
            global_index += 1

    frame = pd.DataFrame(rows).sort_values(["timestamp", "sequence_id", "window_index"]).reset_index(drop=True)
    return frame


def save_synthetic_frame(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


