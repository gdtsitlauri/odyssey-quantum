# Odyssey Quantum Track

The Odyssey repository now contains a dedicated `odyssey.quantum` package in addition to the original hybrid IDS model.

## Scope

The quantum track is split into two layers:

1. Foundations
- single-qubit superposition
- Bell-state entanglement
- GHZ-state preparation
- simple depolarizing-mixture study
- backend/toolchain availability reporting

2. Algorithms
- Deutsch-Jozsa classification
- Bernstein-Vazirani hidden-string recovery
- Grover search amplification
- VQE on a small 2-qubit molecular-style Hamiltonian
- QAOA on triangle MaxCut
- toy Shor order-finding arithmetic walk-through for `N=15`

## Why It Matters

This keeps Odyssey honest about what it is:

- an applied quantum-security repository through `Odyssey-Risk`
- plus a compact, reproducible quantum computing track that demonstrates
  circuit foundations, algorithm families, optimization-style quantum workflows,
  and a toy factorization pathway

## Commands

```powershell
odyssey quantum-foundations --config configs/quantum_foundations.yaml
odyssey quantum-algorithms --config configs/quantum_algorithms.yaml
odyssey quantum-suite --config configs/quantum_suite.yaml
```

## Outputs

By default the suite writes to `results/quantum_suite/`:

- `tables/` for CSV exports
- `reports/` for JSON and Markdown summaries
- `figures/` for compact visual summaries

## Notes

- The track is intentionally laptop-feasible and deterministic.
- It does not claim fault-tolerant scale or hardware quantum advantage.
- The `Shor` component is explicitly a toy arithmetic/order-finding reference, not a full scalable implementation.
