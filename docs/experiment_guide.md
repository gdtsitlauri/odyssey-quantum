# Experiment Guide

## Presets

- `synthetic_small_debug`: fastest smoke-test preset
- `synthetic_research_main`: main synthetic research preset
- `baseline_suite`: strong classical baselines
- `ablation_suite`: ablations of the proposed method
- `seed_stability_suite`: repeated runs across seeds

## Recommended First Run

```powershell
odyssey run-all --config configs/synthetic_small.yaml
```

This produces:

- a metrics table,
- confusion matrix,
- ROC and PR curves,
- calibration plot,
- a markdown experiment report.

## Verified Commands

These commands were exercised on the current repository state with `PYTHONPATH=src` during implementation:

```powershell
odyssey generate-synthetic --config configs/synthetic_small.yaml --output data/processed/smoke.csv
odyssey run-all --config configs/synthetic_small.yaml
odyssey run-odyssey --config configs/public_unsw_smoke.yaml
odyssey run-baselines --config configs/public_unsw_baseline.yaml
odyssey run-odyssey --config configs/public_unsw_odyssey.yaml
odyssey run-odyssey --config configs/public_unsw_odyssey_optimized.yaml
odyssey run-odyssey --config configs/public_unsw_odyssey_aggressive.yaml
odyssey run-ablations --config configs/ablation_suite.yaml
odyssey run-odyssey --config configs/seed_stability_suite.yaml
odyssey run-odyssey --config configs/synthetic_quantum_smoke.yaml
odyssey make-figures --report outputs/reports/latest_metrics.json
odyssey export-paper-assets --source outputs
pytest -q
```

## Final Assessment

See [final_assessment.md](../outputs/reports/final_assessment.md) for the current evidence-based conclusion after the leakage fix and the completed public, ablation, and seed-stability runs.

## Exact Preset Commands

```powershell
odyssey run-odyssey --config configs/synthetic_small.yaml
odyssey run-odyssey --config configs/synthetic_quantum_smoke.yaml
odyssey run-odyssey --config configs/synthetic_research.yaml
odyssey run-baselines --config configs/baseline_suite.yaml
odyssey run-baselines --config configs/public_unsw_baseline.yaml
odyssey run-odyssey --config configs/public_unsw_odyssey.yaml
odyssey run-odyssey --config configs/public_unsw_odyssey_optimized.yaml
odyssey run-odyssey --config configs/public_unsw_odyssey_aggressive.yaml
odyssey run-baselines --config configs/public_unsw_odyssey_ensemble.yaml
odyssey run-ablations --config configs/ablation_suite.yaml
odyssey run-odyssey --config configs/seed_stability_suite.yaml
```

## Quantum Dependency Behavior

- If PennyLane and Qiskit Aer are available, the quantum uncertainty head uses them.
- If `pennylane-qiskit` is also available, backend `auto` prefers `qiskit.aer`.
- If they are unavailable and quantum is not required, Odyssey falls back to a deterministic classical uncertainty surrogate and records the fallback in run metadata.
- If `require_quantum: true`, the run fails with an explicit dependency error.

## Current Best Standalone Odyssey Preset

For the current repository state, the strongest standalone Odyssey result is the aggressive preset:

```powershell
odyssey run-odyssey --config configs/public_unsw_odyssey_aggressive.yaml
```

This preset keeps the public-data fragility path disabled, warm-starts the raw-feature linear probe from logistic regression, and lets validation choose the best post-hoc blend between the attack and final risk logits. In the current evidence, this closes most of the gap to `LogisticRegression`, but `RandomForest` remains the strongest public baseline on PR-AUC.

## Final Best-Effort Benchmark

If you want the strongest last benchmark-oriented run currently implemented:

```powershell
odyssey run-baselines --config configs/public_unsw_odyssey_ensemble.yaml
```

This trains a validation-stacked ensemble over `LogisticRegression`, `RandomForest`, and `Odyssey-Risk`. It is the strongest Odyssey-family public benchmark path currently implemented: it beats standalone Odyssey and LogisticRegression on PR-AUC, but in the present run it still lands just below `RandomForest`.

