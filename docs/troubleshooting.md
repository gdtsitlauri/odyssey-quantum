# Troubleshooting

## Quantum dependencies missing

Install optional quantum packages:

```powershell
pip install -e ".[quantum]"
```

If you want to proceed without them, ensure the config sets `model.require_quantum: false`.

## Slow runs on CPU

- Use `configs/synthetic_small.yaml`.
- Use `configs/synthetic_quantum_smoke.yaml` for the smallest real quantum verification run.
- Reduce `training.epochs`.
- Disable sequence mode.
- Reduce `data.synthetic.n_samples`.
- For public data, reduce `data.public.max_rows`.

## No public dataset found

Place UNSW-NB15 CSV files under `data/raw/unsw_nb15/`. See [data/README.md](/c:/Users/TOM/Desktop/QQ/data/README.md).

## Windows path issues

All internal paths use `pathlib.Path`. Prefer quoted paths when invoking scripts manually from PowerShell.
