| quantity | value |
|---|---|
| parquet_rows | 61348 |
| parquet_last_review_date | 2026-07-10 |
| tracked_in_git | True |
| diff_vs_HEAD | (none) |
| cloud_branch_rows | 29531 |
| cloud_branch_morning | 2524 |
| share_with_factor_ge_2500 | 0.762 |
| share_with_factor_zero | 0.06827 |
| share_capped_60s | 0.04768 |
| share_time_zero | 0 |

Cloud branch (what the deployed dashboard computes):

| learner | n | theta | ATE ms | 95% CI low | 95% CI high | p iid | p cluster(card) |
|---|---|---|---|---|---|---|---|
| ridge | 29531 | 0.06263 | 332.1 | 144.9 | 525.9 | 0.0004212 | 0.0005385 |
| gbm | 29531 | 0.103 | 557.6 | 356 | 766.7 | 2.109e-08 | 3.235e-08 |

Local branch (what was reported):

| learner | n | theta | ATE ms | branch |
|---|---|---|---|---|
| ridge | 4763 | 0.138 | 766.7 | local (hard reviews, as reported) |
| gbm | 4763 | 0.1634 | 920 | local (hard reviews, as reported) |
