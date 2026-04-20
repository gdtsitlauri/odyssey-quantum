# Data Guide

## Synthetic Benchmark

The synthetic benchmark is the default reproducible path. It simulates:

- benign traffic,
- conventional attacks,
- stealthy low-and-slow attacks,
- noisy perturbations,
- post-quantum migration fragility conditions,
- harvest-now-decrypt-later style metadata risk,
- crypto-suite mismatch patterns,
- degraded observability and uncertainty spikes.

Synthetic data is always labeled synthetic in output manifests and reports.

Example generation command:

```powershell
odyssey generate-synthetic --config configs/synthetic_small.yaml --output data/processed/synthetic_small.csv
```

## Public Data: UNSW-NB15

Manual placement is required. Create:

```text
data/raw/unsw_nb15/
```

Place the official CSV files in that directory, or directly under `data/raw/`. The adapter prefers:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

If those files exist, they are used first because they include stable headers. The larger `UNSW-NB15_1.csv` ... `UNSW-NB15_4.csv` files are ignored unless they can be validated as headered inputs. Schema mapping is conservative and only uses columns that are present.

With `use_official_split: true`, Odyssey now keeps the official train/test separation intact and only carves validation rows from the official training CSV. This is the recommended public benchmark mode.

If timestamp-like fields are absent, the pipeline falls back to stratified random splitting.

Public IDS datasets do not expose post-quantum transition fields. Odyssey augments such data with transparent synthetic fragility metadata derived from observed transport and service attributes. These augmented fields are assumptions for research, not measured ground truth.

## Directory Notes

- `data/raw/` is ignored by git except for `.gitkeep`.
- `data/processed/` stores explicitly generated artifacts only.
- Training can also generate synthetic data in memory without persisting to disk.
- Third-party dataset files should not be committed or redistributed through this repository unless their upstream license explicitly allows it.

