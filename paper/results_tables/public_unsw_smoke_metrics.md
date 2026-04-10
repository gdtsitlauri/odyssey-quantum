# public_unsw_smoke_metrics

| model_name | seed | accuracy | precision | recall | f1 | balanced_accuracy | brier_score | ece | recall_at_fixed_fpr | latency_ms_per_sample | parameter_count | training_time_s | tn | fp | fn | tp | roc_auc | pr_auc | notes | uncertainty_mode | backend_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| odyssey_risk | 13 | 0.9008333333333334 | 0.8884892086330936 | 0.9661016949152542 | 0.9256714553404124 | 0.8756605472266803 | 0.0614467873943044 | 0.0263549059467609 | 0.8415906127770535 | 0.0263949583332608 | 19544.0 | 2.3135745999998107 | 680.0 | 186.0 | 52.0 | 1482.0 | 0.972188515285552 | 0.9810338048523426 | blend_weight_risk=0.70,temperature=1.00,val_pr_auc=0.9875 | zero | zero |
