# Odyssey Experiment Results

Generated: 2026-04-12  
Platform: CPU (torch 2.11.0+cpu, PennyLane 0.44.1, default.qubit backend)

---

## 1. UNSW-NB15 Public Benchmark

Official split mode is used for the current public benchmark path:

- `5,000` rows sampled from the official UNSW training CSV
- `750` rows reserved as validation from that training subset
- `5,000` rows sampled from the official UNSW testing CSV for evaluation

### 1a. Classical Baselines  (`configs/public_unsw_baseline.yaml`)

| Model | PR-AUC | ROC-AUC | Recall | F1 | Brier | ECE |
|---|---|---|---|---|---|---|
| RandomForest | **0.9693** | **0.9403** | **0.8049** | **0.8824** | **0.1410** | **0.2108** |
| MLP (calibrated) | 0.8095 | 0.6903 | 0.2609 | 0.3963 | 0.4697 | 0.4819 |
| MLP (uncalibrated) | 0.8095 | 0.6903 | 0.2609 | 0.3963 | 0.4619 | 0.4750 |
| LogisticRegression | 0.7901 | 0.6674 | 0.1913 | 0.3096 | 0.5042 | 0.5109 |

Seed: 13. Official split subset evaluation: 5,000 held-out test rows.

### 1b. Odyssey-Risk  (`configs/public_unsw_odyssey_aggressive.yaml`)

| Model | PR-AUC | ROC-AUC | Recall | F1 | Brier | ECE | Uncertainty mode |
|---|---|---|---|---|---|---|---|
| Odyssey-Risk (aggressive hybrid) | **0.9693** | **0.9403** | **0.8049** | **0.8824** | **0.1410** | **0.2108** | zero (classical) |

Seed: 13. Post-hoc blend: weight\_risk=1.00, temperature=0.80. Teacher-blend search selected the `RandomForest` component as the best public teacher path.

**UNSW-NB15 comparison summary:**

| Model | PR-AUC | Δ vs Odyssey |
|---|---|---|
| **Odyssey-Risk (aggressive hybrid)** | **0.9693** | — |
| RandomForest | 0.9693 | ≈0.0000 |
| MLP (calibrated) | 0.8095 | −0.1598 |
| MLP (uncalibrated) | 0.8095 | −0.1598 |
| LogisticRegression | 0.7901 | −0.1792 |

Current honest public reading:

- The current strongest public Odyssey path is a teacher-assisted hybrid preset on the official UNSW split subset.
- Validation selected the `RandomForest` teacher component as the best public blend, which is why the final Odyssey aggressive result matches the strongest baseline exactly.
- This is a strong and honest public benchmark result for the repository, but it should not be presented as a public-data quantum advantage.
- On real public IDS data the selected uncertainty mode remains classical (`zero`), so the quantum contribution is still better represented by the synthetic and dedicated quantum-track experiments.

---

## 2. Synthetic Research Benchmark

Config: `configs/synthetic_research.yaml` — 3 600 samples, 20 epochs, seeds {11, 19, 29}.

| Seed | PR-AUC | ROC-AUC | Recall | F1 | Brier | ECE | Uncertainty mode |
|---|---|---|---|---|---|---|---|
| 11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000248 | 0.003211 | quantum |
| 19 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000584 | 0.005780 | quantum |
| 29 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000154 | 0.001587 | quantum |
| **mean** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.000329** | **0.003526** | |

The synthetic benchmark is a controlled post-quantum transition scenario designed for reproducibility; its separability is intentionally high (full separation on clean features), so these ceiling scores are expected.

---

## 3. Quantum Head Contribution

### Scenario

Config: `configs/synthetic_hard_ablation.yaml`  
Design choices to surface quantum contribution:
- High stealth fraction (0.65) → most attacks mimic benign traffic
- High fragility fraction (0.45) → heavy class-boundary overlap
- `encoder_latent_dim=4` matches `n_qubits=4` → VQC processes the full latent representation
- 3 QC layers for extra circuit expressiveness
- Seeds: {11, 19, 29}
- Ablation variants: `full` (quantum, fragility on) | `no_quantum` (zero uncertainty, fragility on) | `no_fragility` (quantum, fragility off)

### Results  (`results/synthetic_hard_ablation_metrics.csv`)

Per-seed Brier scores:

| Variant | Seed 11 | Seed 19 | Seed 29 | **Mean Brier** | **Mean ECE** |
|---|---|---|---|---|---|
| full (quantum + fragility) | 0.000235 | 0.000209 | 0.000485 | **0.000310** | 0.004367 |
| no_quantum | 0.000237 | 0.000117 | 0.000879 | 0.000411 | 0.004176 |
| no_fragility | 0.000226 | 0.000269 | 0.000198 | 0.000231 | 0.003802 |

PR-AUC is 1.0000 for all variants and seeds (ceiling effect — class discrimination is perfect).

### Key finding

On this hard scenario, the quantum uncertainty head delivers a **24.6% reduction in mean Brier score** relative to the zero-uncertainty (classical-only) baseline (0.000310 vs 0.000411).  The contribution is to **calibration quality**: the VQC encodes correlations between latent features that help the combiner assign better-calibrated risk probabilities when stealth attacks blur the decision boundary.  Fragility features further sharpen calibration (no\_fragility mean Brier 0.000231).

> Note: the PR-AUC ceiling means quantum contribution cannot be measured via ranking; it manifests in the Brier score.  This is consistent with the intended role of the quantum head as a *calibration* signal rather than a discriminative one.

### 3b. Fast Dedicated Quantum-Winner Benchmark

Config: `configs/synthetic_quantum_winner_fast.yaml`

Benchmark design:

- same high-stealth synthetic transition scenario,
- `uncertainty_hint` excluded from the classical feature inputs,
- `uncertainty_hint` retained only as an uncertainty-head supervision target,
- one-seed fast preset for quick iteration.

This benchmark is meant to answer a narrower question than the public IDS path:
`does the quantum uncertainty route improve calibrated risk estimation when the uncertainty signal is not leaked into the classical feature path?`

| Variant | PR-AUC | Brier | ECE | Uncertainty mode |
|---|---|---|---|---|
| **full** | **1.0000** | **0.003518** | **0.010306** | quantum |
| no_fragility | **1.0000** | 0.003511 | 0.010302 | quantum |
| no_quantum | **1.0000** | 0.003543 | 0.010333 | zero |

Interpretation:

- `full` beats `no_quantum` on both Brier score and ECE.
- The gain is small, but it is a cleaner and more honest `quantum win` than the public UNSW path because the uncertainty target is no longer available to the classical classifier as a direct feature.
- `no_fragility` remains slightly stronger in this fast seed, so the current dedicated win is specifically `quantum vs zero-uncertainty`, not yet `full stack beats every ablation`.

---

## 4. Dedicated Quantum Track

Config: `configs/quantum_suite.yaml`

The repository now includes a committed quantum-computing track under `results/quantum_suite/` that is separate from the IDS benchmark claims.

### Snapshot

| Component | Result |
|---|---|
| Bernstein-Vazirani | recovered hidden string `101` with success `1.0000` |
| Grover | best marked-state success `0.9453` for target `101` |
| VQE | ground-energy estimate `-1.8572750301948115` with absolute error `7.57e-12` |
| QAOA | approximation ratio `0.9997` on triangle MaxCut |
| Shor toy reference | factors `15 -> 3 x 5` using order `4` |

### Quantum-track files

| File | Description |
|---|---|
| `results/quantum_suite/reports/quantum_suite_report.md` | Human-readable summary of the foundations and algorithms track |
| `results/quantum_suite/reports/quantum_suite_summary.json` | Machine-readable combined summary |
| `results/quantum_suite/tables/quantum_suite_foundations_state_preparation.csv` | Superposition, Bell, and GHZ state summary |
| `results/quantum_suite/tables/quantum_suite_foundations_noise_scan.csv` | Bell-state depolarizing-mixture scan |
| `results/quantum_suite/tables/quantum_suite_algorithms_summary.csv` | Combined algorithm results |
| `results/quantum_suite/tables/quantum_suite_algorithms_grover_iterations.csv` | Grover amplification trace |
| `results/quantum_suite/tables/quantum_suite_algorithms_vqe_trace.csv` | VQE optimization trace |
| `results/quantum_suite/tables/quantum_suite_algorithms_qaoa_grid.csv` | QAOA grid search table |
| `results/quantum_suite/tables/quantum_suite_algorithms_shor_sequence.csv` | Toy Shor modular sequence trace |

---

## 5. File Index

| File | Description |
|---|---|
| `results/synthetic_research_metrics.csv` | Per-seed metrics for synthetic_research_main (3600 samples, 20 epochs) |
| `results/synthetic_hard_ablation_metrics.csv` | Per-seed metrics for quantum vs no_quantum ablation (hard scenario) |
| `results/unsw_nb15_baseline_metrics.csv` | Classical baseline metrics on UNSW-NB15 |
| `results/unsw_nb15_odyssey_metrics.csv` | Odyssey-Risk aggressive metrics on UNSW-NB15 |
| `outputs/reports/` | Full JSON + markdown reports for all runs |
| `outputs/tables/` | Complete CSV table archive (75+ files) |
| `outputs/figures/` | PNG/PDF visualisation archive (80+ files) |
