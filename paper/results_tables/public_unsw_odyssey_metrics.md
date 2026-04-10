# public_unsw_odyssey_metrics

| model_name | seed | accuracy | precision | recall | f1 | balanced_accuracy | brier_score | ece | recall_at_fixed_fpr | latency_ms_per_sample | parameter_count | training_time_s | tn | fp | fn | tp | roc_auc | pr_auc | notes | uncertainty_mode | backend_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| odyssey_risk | 13 | 0.88 | 0.90625 | 0.90625 | 0.90625 | 0.8697916666666667 | 0.0755556252153626 | 0.0620363592707871 | 0.8125 | 8.382647500000076 | 8957.0 | 57.21503699999994 | 60.0 | 12.0 | 12.0 | 116.0 | 0.9618055555555556 | 0.9786488291316496 | blend_weight_risk=0.00,temperature=1.15,val_pr_auc=0.9804 | quantum | default.qubit |
