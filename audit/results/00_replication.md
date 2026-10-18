| analysis | quantity | reported | reproduced | match | source |
|---|---|---|---|---|---|
| PSM v2 | hard reviews (ef<2.5) | 4252 | 4252 | PASS | psm_response_time_v2_executed.ipynb cell 2 |
| PSM v2 | morning hard reviews | 319 | 319 | PASS | psm_response_time_v2_executed.ipynb cell 2 |
| PSM v2 | evening hard reviews | 3933 | 3933 | PASS | psm_response_time_v2_executed.ipynb cell 2 |
| PSM v2 | capped at 60s | 214 | 214 | PASS | psm_response_time_v2_executed.ipynb cell 2 |
| PSM v2 | after excluding capped | 4038 | 4038 | PASS | psm_response_time_v2_executed.ipynb cell 4 |
| PSM v2 | morning after exclusion | 309 | 309 | PASS | psm_response_time_v2_executed.ipynb cell 4 |
| PSM v2 | PS model train accuracy | 0.923 | 0.923477 | PASS | psm_response_time_v2_executed.ipynb cell 6 |
| PSM v2 | PS coef ef | -0.0125 | -0.012489 | PASS | psm_response_time_v2_executed.ipynb cell 6 |
| PSM v2 | PS coef log_ivl | 0.0336 | 0.03356 | PASS | psm_response_time_v2_executed.ipynb cell 6 |
| PSM v2 | PS coef log_days_since | 0.0447 | 0.044737 | PASS | psm_response_time_v2_executed.ipynb cell 6 |
| PSM v2 | PS coef month_idx | 0.031 | 0.031047 | PASS | psm_response_time_v2_executed.ipynb cell 6 |
| PSM v2 | PS mean, evening | 0.0765 | 0.07648 | PASS | psm_response_time_v2_executed.ipynb cell 6 |
| PSM v2 | PS mean, morning | 0.077 | 0.076972 | PASS | psm_response_time_v2_executed.ipynb cell 6 |
| PSM v2 | PS max, evening | 0.0974 | 0.097443 | PASS | psm_response_time_v2_executed.ipynb cell 6 |
| PSM v2 | PS max, morning | 0.094 | 0.093968 | PASS | psm_response_time_v2_executed.ipynb cell 6 |
| PSM v2 | PS min, evening | 0.0603 | 0.06029 | PASS | psm_response_time_v2_executed.ipynb cell 6 |
| PSM v2 | caliper | 0.01745 | 0.017447 | PASS | psm_response_time_v2_executed.ipynb cell 7 |
| PSM v2 | matched pairs | 309 | 309 | PASS | psm_response_time_v2_executed.ipynb cell 7 |
| PSM v2 | matched median, morning (ms) | 5497 | 5497 | PASS | psm_response_time_v2_executed.ipynb cell 7 |
| PSM v2 | matched median, evening (ms) | 4413 | 4413 | PASS | psm_response_time_v2_executed.ipynb cell 7 |
| PSM v2 | ATE mean diff (ms) | 508 | 507.583 | PASS | psm_response_time_v2_executed.ipynb cell 12 |
| PSM v2 | ATE median diff (ms) | 1084 | 1084 | PASS | psm_response_time_v2_executed.ipynb cell 12 |
| PSM v2 | mean-diff CI low | -858 | -857.689 | PASS | psm_response_time_v2_executed.ipynb cell 12 |
| PSM v2 | mean-diff CI high | 1844 | 1844.2 | PASS | psm_response_time_v2_executed.ipynb cell 12 |
| PSM v2 | median-diff CI low | 209 | 208.975 | PASS | psm_response_time_v2_executed.ipynb cell 12 |
| PSM v2 | median-diff CI high | 1924 | 1924 | PASS | psm_response_time_v2_executed.ipynb cell 12 |
| PSM v2 | t-test p | 0.4979 | 0.497943 | PASS | psm_response_time_v2_executed.ipynb cell 12 |
| PSM v2 | Mann-Whitney U | 54044 | 54044 | PASS | psm_response_time_v2_executed.ipynb cell 12 |
| PSM v2 | Mann-Whitney p | 0.0045 | 0.00451 | PASS | psm_response_time_v2_executed.ipynb cell 12 |
| PSM v2 | SMD after matching: ef | 0.014531 | 0.014531 | PASS | psm_response_time_v2_executed.ipynb cell 8 |
| PSM v2 | SMD after matching: log_ivl | 0.032767 | 0.032767 | PASS | psm_response_time_v2_executed.ipynb cell 8 |
| PSM v2 | SMD after matching: log_days_since | 0.095352 | 0.095352 | PASS | psm_response_time_v2_executed.ipynb cell 8 |
| PSM v2 | SMD after matching: month_idx | 0.116792 | 0.116792 | PASS | psm_response_time_v2_executed.ipynb cell 8 |
| PSM Phase 3 | PSM dataset rows | 21803 | 21803 | PASS | anki_causal_analysis_executed.ipynb cell 14 |
| PSM Phase 3 | morning rows | 1375 | 1375 | PASS | anki_causal_analysis_executed.ipynb cell 14 |
| PSM Phase 3 | evening rows | 20428 | 20428 | PASS | anki_causal_analysis_executed.ipynb cell 14 |
| PSM Phase 3 | PS model train accuracy | 0.937 | 0.936935 | PASS | anki_causal_analysis_executed.ipynb cell 15 |
| PSM Phase 3 | caliper | 0.0306 | 0.030613 | PASS | anki_causal_analysis_executed.ipynb cell 17 |
| PSM Phase 3 | matched pairs | 1375 | 1375 | PASS | anki_causal_analysis_executed.ipynb cell 17 |
| PSM Phase 3 | ATE (retention) | -0.0007 | -0.000727 | PASS | anki_causal_analysis_executed.ipynb cell 19 |
| PSM Phase 3 | t statistic | -0.219 | -0.218978 | PASS | anki_causal_analysis_executed.ipynb cell 19 |
| PSM Phase 3 | t-test p | 0.8267 | 0.826683 | PASS | anki_causal_analysis_executed.ipynb cell 19 |
| PSM Phase 3 | bootstrap CI low | -0.0065 | -0.006545 | PASS | anki_causal_analysis_executed.ipynb cell 19 |
| PSM Phase 3 | bootstrap CI high | 0.0051 | 0.005091 | PASS | anki_causal_analysis_executed.ipynb cell 19 |
| DML | hard reviews | 4763 | 4763 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML | morning | 530 | 530 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML | evening | 4233 | 4233 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML ridge | theta (log RT) | 0.138 | 0.137956 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML ridge | SE | 0.0344 | 0.034421 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML ridge | ATE (ms) | 767 | 766.685 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML ridge | CI low (ms) | 379 | 378.532 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML ridge | CI high (ms) | 1182 | 1181.93 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML gbm | theta (log RT) | 0.1634 | 0.163393 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML gbm | ATE (ms) | 920 | 919.964 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML gbm | CI low (ms) | 511 | 510.531 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |
| DML gbm | CI high (ms) | 1359 | 1358.84 | PASS | DML run of 2026-07-10 (conversation log) / dashboard.py run_dml() |

56 of 56 checks reproduce within the printed precision.
