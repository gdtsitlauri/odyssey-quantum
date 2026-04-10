# Odyssey

Odyssey is a publication-oriented research repository for studying quantum-resilient, risk-aware intrusion detection during post-quantum cryptographic transition. The repository implements a hybrid classical-quantum research prototype, strong classical baselines, ablations, reproducible experiment automation, and paper-support assets.

## Why This Matters

By 2026 and beyond, defenders will face mixed cryptographic environments where legacy key exchange, transitional post-quantum deployments, observability gaps, and long-horizon data exposure all interact. Odyssey studies whether intrusion detection can improve rare-attack sensitivity and confidence calibration by combining:

- classical attack likelihood estimation,
- a small quantum uncertainty head,
- a post-quantum transition fragility score,
- temporal stability across event windows.

Odyssey is a research prototype. It is not a production security product and does not claim hardware quantum advantage.

## Architecture

```mermaid
flowchart LR
    A[Raw or Synthetic Event Windows] --> B[Preprocessing and Feature Builder]
    B --> C[Classical Encoder]
    C --> D[Attack Head]
    C --> E[Latent Bottleneck]
    E --> F[Quantum Uncertainty Head]
    B --> G[Fragility Scorer]
    D --> H[Risk Combiner]
    F --> H
    G --> H
    H --> I[Composite Training Loss]
    H --> J[Metrics, Tables, Figures, Reports]
```

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Optional quantum dependencies:

```powershell
pip install -e ".[quantum]"
```

Bash or WSL:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quickstart

Generate the smallest synthetic benchmark:

```powershell
odyssey generate-synthetic --config configs/synthetic_small.yaml --output data/processed/synthetic_small.csv
```

Run the baseline suite:

```powershell
odyssey run-baselines --config configs/baseline_suite.yaml
```

Run Odyssey-Risk on the research preset:

```powershell
odyssey run-odyssey --config configs/synthetic_research.yaml
```

Run the strongest current public Odyssey preset:

```powershell
odyssey run-odyssey --config configs/public_unsw_odyssey_aggressive.yaml
```

Run the smallest real quantum verification preset:

```powershell
odyssey run-odyssey --config configs/synthetic_quantum_smoke.yaml
```

Run the full experiment stack:

```powershell
odyssey run-all --config configs/synthetic_small.yaml
```

## Data Options

- Synthetic benchmark: fully reproducible and transparent. Recommended for first runs.
- UNSW-NB15 adapter: place files under `data/raw/unsw_nb15/` or directly under `data/raw/` and follow [data/README.md](/c:/Users/TOM/Desktop/QQ/data/README.md).

## GitHub Publishing Notes

- Raw datasets are not committed. `data/raw/` stays gitignored by design.
- Generated experiment artifacts under `outputs/` stay gitignored by default.
- If you want to publish selected results, export only the specific tables or figures you want to keep and commit them intentionally.
- Do not re-upload third-party datasets unless their original license explicitly allows redistribution.

## Experiment Commands

- `odyssey generate-synthetic --config configs/synthetic_small.yaml --output data/processed/synthetic_small.csv`
- `odyssey run-baselines --config configs/baseline_suite.yaml`
- `odyssey run-odyssey --config configs/synthetic_research.yaml`
- `odyssey run-baselines --config configs/public_unsw_baseline.yaml`
- `odyssey run-odyssey --config configs/public_unsw_odyssey.yaml`
- `odyssey run-odyssey --config configs/public_unsw_odyssey_optimized.yaml`
- `odyssey run-odyssey --config configs/public_unsw_odyssey_aggressive.yaml`
- `odyssey run-baselines --config configs/public_unsw_odyssey_ensemble.yaml`
- `odyssey run-odyssey --config configs/synthetic_quantum_smoke.yaml`
- `odyssey run-ablations --config configs/ablation_suite.yaml`
- `odyssey run-all --config configs/synthetic_small.yaml`
- `odyssey make-figures --report outputs/reports/latest_metrics.json`
- `odyssey export-paper-assets --source outputs`

## Expected Outputs

Runs write to `outputs/`:

- `outputs/tables/` for metrics CSVs and aggregated summaries
- `outputs/figures/` for PNG and PDF figures
- `outputs/logs/` for run logs and config snapshots
- `outputs/reports/` for markdown reports and JSON summaries

## Limitations

- Quantum evaluation is simulator-based and intentionally small to remain laptop-feasible.
- Post-quantum transition fragility features on public IDS data are augmentation assumptions, not observed labels.
- Public dataset support is intentionally conservative in v1 and centers on UNSW-NB15.
- Default experiments are tuned for CPU feasibility, not maximum benchmark performance.
- Real `qiskit.aer` runs are substantially slower than the classical fallback on a laptop CPU.
- On the current `UNSW-NB15` adapter, the strongest public Odyssey preset uses the classical/zero uncertainty path; the quantum head is still useful primarily as a research component on synthetic and quantum-smoke runs.
- A validation-stacked `Odyssey + RandomForest + LogisticRegression` ensemble is included as a final best-effort benchmark path, but on the current public run it still remains marginally below `RandomForest` on PR-AUC.

## Ethical Note

This repository is intended for defensive security research and reproducible scientific study. Synthetic scenarios that mimic stealth or migration fragility are included to help defenders reason about failure modes, not to support offensive deployment.

## License

This repository is released under the MIT License. See [LICENSE](/c:/Users/TOM/Desktop/QQ/LICENSE).

## Extending the Framework

- Add new public data adapters under `src/odyssey/data/`
- Add alternative fragility heuristics under `src/odyssey/features/`
- Add stronger temporal encoders or calibration methods under `src/odyssey/models/` and `src/odyssey/training/`
- Update experiment presets under `configs/`
- Export new paper artifacts through `scripts/export_paper_assets.py`

## Fastest Path To First Results

1. Install the base dependencies.
2. Run `odyssey run-all --config configs/synthetic_small.yaml`.
3. Inspect `outputs/reports/synthetic_small_debug_all_report.md` and the figures in `outputs/figures/`.

