# public_unsw_odyssey_aggressive_metrics

| model_name | seed | accuracy | precision | recall | f1 | balanced_accuracy | brier_score | ece | recall_at_fixed_fpr | latency_ms_per_sample | parameter_count | training_time_s | tn | fp | fn | tp | roc_auc | pr_auc | notes | uncertainty_mode | backend_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| odyssey_risk | 13 | 0.885 | 0.8947368421052632 | 0.9296875 | 0.9118773946360154 | 0.8676215277777778 | 0.0749896367317362 | 0.0512470669817412 | 0.8125 | 0.0225219999992987 | 13564.0 | 0.3327842999997301 | 58.0 | 14.0 | 9.0 | 119.0 | 0.9620225694444444 | 0.9787384949961442 | blend_weight_risk=1.00,temperature=1.00,val_pr_auc=0.9809 | zero | zero |
