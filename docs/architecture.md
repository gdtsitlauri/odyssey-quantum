# Odyssey Architecture And Engineering Log

## 2026-04-10

### Phase 1

- Initialized the repository as a greenfield research codebase.
- Chose `argparse` for CLI stability and lower dependency weight.
- Locked the first public-data adapter to `UNSW-NB15`.
- Set Python compatibility to `3.11+` so the project remains faithful to the requested stack while still working on the detected Python `3.12` environment.
- Established `outputs/` as the only artifact sink.

### Phase 2

- Implemented the synthetic transition benchmark generator with transparent attack, stealth, noise, and fragility scenarios.
- Added the `UNSW-NB15` public adapter with manual placement instructions and augmentation manifests for transition metadata.
- Added deterministic preprocessing, group-aware splitting, and optional sequence packing.

### Phase 3

- Implemented logistic regression, random forest, MLP, calibrated MLP, and GRU baselines.
- Added deterministic training utilities, device resolution, and early stopping on validation PR-AUC.

### Phase 4

- Implemented the Odyssey-Risk model with a classical encoder, attack head, fragility integration, and configurable uncertainty head modes.
- Added a PennyLane-based quantum path plus documented classical fallback and ablation heads.
- Added trainable positive combiner weights and saved normalization statistics for fragility and uncertainty.

### Phase 5

- Implemented focal, Brier-like, temporal consistency, and minority-margin loss terms.
- Added metrics, ROC/PR/calibration/confusion plots, runtime summaries, CSV exports, and markdown reporting.

### Phase 6

- Added CLI orchestration for synthetic generation, baselines, Odyssey, combined runs, ablations, figure regeneration, and paper-asset export.
- Added config-driven presets for debug, research, baselines, ablations, and seed stability.

### Phase 7

- Added `paper/` support files, generated markdown table export, and a figure manifest for manuscript assembly.
- Wrote README, setup scripts, troubleshooting notes, and the experiment guide.

### Phase 8

- Added unit tests for the data pipeline, models, losses, metrics, and CLI smoke execution.
- Verified a real CPU smoke run on `configs/synthetic_small.yaml` through `run-all`.
- Verified figure regeneration and paper-asset export commands against generated outputs.
- Added robust `UNSW-NB15` discovery for both `data/raw/unsw_nb15/` and direct `data/raw/` placement.
- Added public-data presets and verified public baseline plus Odyssey runs against the provided dataset files.
- Installed and verified the real quantum path with PennyLane + `pennylane-qiskit` + `qiskit.aer`, including a dedicated `synthetic_quantum_smoke` preset.
- Fixed a public-data label leakage bug by excluding `attack_cat` from the feature pipeline when it is the source of the binary label.
- Completed matched public comparison, synthetic ablation, and seed-stability runs with laptop-feasible presets.
- Wrote a final assessment in `outputs/reports/final_assessment.md` documenting that the repository is complete but the current hypothesis is not yet strongly supported by the present results.
- Fixed an additional leakage path in public fragility augmentation by removing `attack_cat` from derived metadata generation.
- Strengthened the Odyssey attack head, added auxiliary attack supervision, and changed uncertainty/fragility to act as attack-conditioned modulation terms.
- Added an optimized public Odyssey preset and reran the main comparison suites with the updated model.
- Rebranded the full repository from `QuAegis` to `Odyssey`, including package paths, CLI entrypoint, configs, reports, and generated paper assets.
- Added a raw-feature linear probe inside `Odyssey-Risk`, warm-started it from logistic regression on the training split, and added validation-selected post-hoc blending between `attack_logit` and `risk_logit`.
- Added `configs/public_unsw_odyssey_aggressive.yaml` for a second, more targeted public optimization round.
- Added `configs/public_unsw_odyssey_ensemble.yaml` plus a validation-stacked ensemble over `LogisticRegression`, `RandomForest`, and `Odyssey-Risk` as the final best-effort benchmark extension.
- Regenerated the repository outputs end-to-end under the new Odyssey naming and reran the main public, synthetic, ablation, seed-stability, and quantum-smoke suites.
- Current public conclusion after the second optimization round: Odyssey now closes most of the gap to `LogisticRegression` and surpasses it on recall, F1, and Brier score, but `RandomForest` still leads on PR-AUC for `UNSW-NB15`.

