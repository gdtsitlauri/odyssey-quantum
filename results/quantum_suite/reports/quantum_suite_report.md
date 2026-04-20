# Odyssey Quantum Track

## Summary

- Suite: `quantum_suite`
- Preferred backend: `pennylane+qiskit.aer`
- Backend availability: PennyLane=True | Qiskit=True | Aer=True

## Foundations

- `single_qubit_superposition` top state `0` with probability 0.5000
- `bell_state` top state `00` with probability 0.5000
- `ghz_3q` top state `000` with probability 0.5000

## Algorithms

- `deutsch_jozsa` classified `constant` correctly=True
- `deutsch_jozsa` classified `balanced` correctly=True
- `bernstein_vazirani` recovered `101` from hidden string `101` with success 1.0000
- `grover` best measurement `101` for marked state `101` with success 0.9453
- `vqe` estimated ground energy -1.857275 (error 0.000000)
- `qaoa` reached approximation ratio 0.9997 on `triangle_maxcut`
- `shor_toy_reference` factored 15 into 3 x 5 with order 4

## Artifacts

- `state_preparation`: `results\quantum_suite\tables\quantum_suite_foundations_state_preparation.csv`
- `noise_scan`: `results\quantum_suite\tables\quantum_suite_foundations_noise_scan.csv`
- `summary`: `results\quantum_suite\tables\quantum_suite_algorithms_summary.csv`
- `grover_iterations`: `results\quantum_suite\tables\quantum_suite_algorithms_grover_iterations.csv`
- `vqe_trace`: `results\quantum_suite\tables\quantum_suite_algorithms_vqe_trace.csv`
- `qaoa_grid`: `results\quantum_suite\tables\quantum_suite_algorithms_qaoa_grid.csv`
- `shor_sequence`: `results\quantum_suite\tables\quantum_suite_algorithms_shor_sequence.csv`
