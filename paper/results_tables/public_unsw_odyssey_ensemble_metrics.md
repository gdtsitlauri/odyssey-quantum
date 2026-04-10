# public_unsw_odyssey_ensemble_metrics

| model_name | seed | accuracy | precision | recall | f1 | balanced_accuracy | brier_score | ece | recall_at_fixed_fpr | latency_ms_per_sample | parameter_count | training_time_s | tn | fp | fn | tp | roc_auc | pr_auc | notes | uncertainty_mode | backend_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| odyssey_stacked_ensemble | 13 | 0.915 | 0.9586776859504132 | 0.90625 | 0.931726907630522 | 0.9184027777777778 | 0.0641793812272815 | 0.0784599748612772 | 0.90625 | 0.0593765000030543 | 28685.0 | 0.5929771999999502 | 67.0 | 5.0 | 12.0 | 116.0 | 0.9774305555555556 | 0.9876243488302668 | members=logistic_regression,random_forest,odyssey_risk;meta_coeffs=1.5398,3.4452,1.4299;meta_bias=-3.6735;odyssey_blend_weight=1.00;odyssey_blend_temp=1.00 | zero | zero |
