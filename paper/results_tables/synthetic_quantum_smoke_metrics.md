# synthetic_quantum_smoke_metrics

| model_name | seed | accuracy | precision | recall | f1 | balanced_accuracy | brier_score | ece | recall_at_fixed_fpr | latency_ms_per_sample | parameter_count | training_time_s | tn | fp | fn | tp | roc_auc | pr_auc | notes | uncertainty_mode | backend_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| odyssey_risk | 5 | 0.875 | 0.8 | 1.0 | 0.8888888888888888 | 0.875 | 0.0858834682532763 | 0.2442157641053199 | 1.0 | 84.61640000007264 | 8420.0 | 53.1889441000003 | 3.0 | 1.0 | 0.0 | 4.0 | 1.0 | 1.0 | blend_weight_risk=0.00,temperature=0.85,val_pr_auc=1.0000 | quantum | qiskit.aer[aer_simulator_statevector] |
