# public_unsw_odyssey_optimized_metrics

| model_name | seed | accuracy | precision | recall | f1 | balanced_accuracy | brier_score | ece | recall_at_fixed_fpr | latency_ms_per_sample | parameter_count | training_time_s | tn | fp | fn | tp | roc_auc | pr_auc | notes | uncertainty_mode | backend_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| odyssey_risk | 13 | 0.885 | 0.900763358778626 | 0.921875 | 0.9111969111969112 | 0.8706597222222222 | 0.0749100568175885 | 0.0490313818657887 | 0.8046875 | 8.459422000000814 | 13685.0 | 107.96269889999984 | 59.0 | 13.0 | 10.0 | 118.0 | 0.9613715277777778 | 0.9782081115131506 | blend_weight_risk=1.00,temperature=1.00,val_pr_auc=0.9825 | quantum | default.qubit |
