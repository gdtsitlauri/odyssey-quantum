# ablation_suite_metrics

| model_name | seed | accuracy | precision | recall | f1 | balanced_accuracy | brier_score | ece | recall_at_fixed_fpr | latency_ms_per_sample | parameter_count | training_time_s | tn | fp | fn | tp | roc_auc | pr_auc | notes | uncertainty_mode | backend_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0014710505082308 | 0.0141928225959418 | 1.0 | 8.520797500000299 | 8940.0 | 56.81602120000025 | 88.0 | 0.0 | 0.0 | 32.0 | 1.0 | 1.0 | blend_weight_risk=1.00,temperature=0.85,val_pr_auc=1.0000 | quantum | default.qubit |
| no_fragility | 7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0016732813606782 | 0.0155287441123315 | 1.0 | 8.174240833333593 | 8940.0 | 55.06797710000046 | 88.0 | 0.0 | 0.0 | 32.0 | 1.0 | 1.0 | blend_weight_risk=1.00,temperature=0.85,val_pr_auc=1.0000 | quantum | default.qubit |
| no_quantum | 7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0006269762226347 | 0.0069300501335542 | 1.0 | 0.0206383333306803 | 8819.0 | 0.1432235000002037 | 88.0 | 0.0 | 0.0 | 32.0 | 1.0 | 1.0 | blend_weight_risk=0.00,temperature=0.85,val_pr_auc=1.0000 | zero | zero |
| no_temporal | 7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0006188484728567 | 0.0069104655519292 | 1.0 | 8.687330000005508 | 8940.0 | 34.45915749999949 | 88.0 | 0.0 | 0.0 | 32.0 | 1.0 | 1.0 | blend_weight_risk=0.00,temperature=0.85,val_pr_auc=1.0000 | quantum | default.qubit |
| random_uncertainty | 7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0006427692384468 | 0.0069840846715294 | 1.0 | 0.0199974999986807 | 8819.0 | 0.1556227000000944 | 88.0 | 0.0 | 0.0 | 32.0 | 1.0 | 1.0 | blend_weight_risk=0.00,temperature=0.85,val_pr_auc=1.0000 | random | random |
