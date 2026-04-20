# Quantum Appendix

## Purpose

This appendix documents the dedicated quantum-computing track that accompanies the main Odyssey intrusion-detection repository.

## Included Components

- Foundations: superposition, Bell-state entanglement, GHZ preparation, and a simple noise study
- Algorithms: Deutsch-Jozsa, Bernstein-Vazirani, Grover, VQE, QAOA, and a toy Shor order-finding walk-through
- Applied case study: Odyssey-Risk, where a compact quantum uncertainty head is used inside a defensive IDS research prototype

## Positioning

The main public-security benchmark remains a classical-vs-hybrid IDS study.
The quantum appendix broadens the repository so that it also demonstrates:

- core circuit intuition
- oracle-style algorithms
- variational optimization workflows
- a small factorization-related order-finding pathway

## Reproducibility

Run:

```powershell
odyssey quantum-suite --config configs/quantum_suite.yaml
```

Artifacts are exported under `results/quantum_suite/`.
