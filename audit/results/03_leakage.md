### A1. Change in revlog.factor against the previous row of the same card (review -> review rows)

37,852 consecutive review rows. If revlog.factor were the factor BEFORE the answer, the change would line up with the previous row's button instead.

| ease of THIS review | mean | std | count |
|---|---|---|---|
| 1 | -185.6 | 49.01 | 250 |
| 2 | -146.4 | 22.51 | 6359 |
| 3 | -0.04758 | 5.343 | 6305 |
| 4 | 149.9 | 5.372 | 24938 |

### A1b. Same change, grouped by the PREVIOUS row's button

| ease of PREVIOUS review | mean | std | count |
|---|---|---|---|
| 2 | 20.61 | 130.6 | 7690 |
| 3 | 58.17 | 120.9 | 7126 |
| 4 | 94.95 | 101.9 | 23036 |

### A2. revlog.ivl by button (type 1 = review rows)

ivl is the interval assigned by this answer; every Again row carries ivl = -600 (the 10-minute relearning step, in seconds).

| ease | rows | share_ivl_negative | median_ivl | median_lastIvl |
|---|---|---|---|---|
| 1 | 345 | 1 | -600 | 9 |
| 2 | 8241 | 0 | 14 | 11 |
| 3 | 8217 | 0 | 33 | 14 |
| 4 | 31883 | 0 | 63 | 19 |

### B. PSM v2 sample (ef from revlog.factor < 2.5): status of each sample row's factor BEFORE its own answer

| pre-answer factor status | evening | morning | total | evening % | morning % | total % |
|---|---|---|---|---|---|---|
| NOT hard before this answer | 996 | 74 | 1070 | 26.7 | 23.9 | 26.5 |
| already hard before this answer | 2733 | 235 | 2968 | 73.3 | 76.1 | 73.5 |

### B. DML sample (revlog.factor < 2500): status of each sample row's factor BEFORE its own answer

| pre-answer factor status | evening | morning | total | evening % | morning % | total % |
|---|---|---|---|---|---|---|
| NOT hard before this answer | 1145 | 140 | 1285 | 27.0 | 26.4 | 27.0 |
| already hard before this answer | 3088 | 390 | 3478 | 73.0 | 73.6 | 73.0 |

### B2. Phase 3: outcome `retained` against the covariate `ivl` sign

AUC of log_ivl for predicting retained == 0: 0.9014. log_ivl is log1p(|revlog.ivl|), written after the answer.

| retained | ivl < 0 | ivl > 0 |
|---|---|---|
| retained = 0 (Again) | 184 | 0 |
| retained = 1 | 1659 | 19960 |

### C. Audited DML folds (KFold, shuffle=True, random_state=42, row level): what leaks across the train/test boundary

4,763 rows from 1,885 cards (1,217 notes); reviews per card mean 2.53; cards per note in the collection: {3: 3490}.

| fold | test rows | % test rows whose CARD is in training | % whose NOTE is in training | % with a LATER review of the same card in training | % with an EARLIER review of the same card in training | % of training rows dated after the test fold's first row |
|---|---|---|---|---|---|---|
| 0 | 953 | 80.1 | 89.7 | 58.0 | 55.7 | 99.8 |
| 1 | 953 | 78.2 | 89.1 | 53.7 | 51.5 | 100.0 |
| 2 | 953 | 77.8 | 87.6 | 55.5 | 53.6 | 99.8 |
| 3 | 952 | 78.2 | 88.2 | 52.6 | 56.4 | 100.0 |
| 4 | 952 | 81.4 | 89.6 | 55.9 | 57.7 | 99.9 |

### D. DML estimate under alternative fold designs, with cluster-robust standard errors

Same samples, same learners. Only the folds, the scaling step or the SE changes.

| learner | design | theta | ATE ms | SE iid | SE cluster(card) | SE cluster(note) | p iid | p cluster(card) | p cluster(note) |
|---|---|---|---|---|---|---|---|---|---|
| ridge | audited: row-level KFold, scaler on all rows | 0.138 | 766.7 | 0.03442 | 0.03523 | 0.03522 | 6.127e-05 | 9.018e-05 | 8.957e-05 |
| ridge | row-level KFold, scaler fit inside each training fold | 0.138 | 766.9 | 0.03442 | 0.03524 | 0.03522 | 6.102e-05 | 8.985e-05 | 8.918e-05 |
| ridge | folds grouped by note (no note in both train and test) | 0.138 | 767.2 | 0.03449 | 0.03536 | 0.03538 | 6.279e-05 | 9.46e-05 | 9.539e-05 |
| ridge | folds grouped by card | 0.1375 | 763.7 | 0.03441 | 0.03526 | 0.03527 | 6.461e-05 | 9.692e-05 | 9.72e-05 |
| ridge | 5 contiguous time blocks (test block never overlaps train in time) | 0.05016 | 266.6 | 0.0349 | 0.03601 | 0.03667 | 0.1506 | 0.1636 | 0.1713 |
| gbm | audited: row-level KFold, scaler on all rows | 0.1634 | 920 | 0.03543 | 0.03592 | 0.03548 | 3.996e-06 | 5.404e-06 | 4.123e-06 |
| gbm | row-level KFold, scaler fit inside each training fold | 0.1636 | 921.1 | 0.03542 | 0.03592 | 0.03548 | 3.878e-06 | 5.25e-06 | 4.004e-06 |
| gbm | folds grouped by note (no note in both train and test) | 0.1556 | 872.6 | 0.03574 | 0.03613 | 0.03592 | 1.339e-05 | 1.655e-05 | 1.473e-05 |
| gbm | folds grouped by card | 0.1561 | 875.5 | 0.03572 | 0.03607 | 0.03559 | 1.242e-05 | 1.514e-05 | 1.155e-05 |
| gbm | 5 contiguous time blocks (test block never overlaps train in time) | 0.05821 | 310.7 | 0.03296 | 0.03349 | 0.03361 | 0.07738 | 0.08221 | 0.08327 |

### E. Sensitivity of each estimate to the clock and to point-in-time features

Diagnostics only. The specifications change the sample as well as the covariates, so rows are not like-for-like replacements of the audited run.

| specification | PSM v2: n (morning/evening) | PSM v2 median diff ms (audited order) | PSM v2 MW p (audited order) | PSM v2 median diff across 100 row orders: mean [min, max] | Phase 3 ATE pp | Phase 3 p | DML ridge ATE ms (p cluster) | DML gbm ATE ms (p cluster) | DML n (morning/evening) |
|---|---|---|---|---|---|---|---|---|---|
| S0 audited (UTC clock, post-answer features) | 309/3729 | 1084 | 0.00451 | 1045 [629, 1332] | -0.07273 | 0.8267 | +767 (p 9e-05) | +920 (p 5.4e-06) | 530/4233 |
| S1 point-in-time features (UTC clock) | 301/3463 | 44 | 0.1454 | 44 [44, 44] | -0.1735 | 0.6535 | +643 (p 0.00053) | +795 (p 3.7e-05) | 496/3953 |
| S2 Eastern clock (post-answer features) | 2999/1930 | 636 | 2.2e-10 | 610 [444, 860] | 0.2169 | 0.0256 | +333 (p 0.0057) | +554 (p 3.9e-05) | 4172/2191 |
| S3 Eastern clock + point-in-time features | 2709/1803 | 357 | 6.366e-05 | 357 [357, 357] | 0.704 | 2.55e-09 | +495 (p 3e-05) | +779 (p 5.6e-09) | 3781/2038 |

### F1. Reconstruction of the notebook's quartile table from revlog (cutoff 2026-04-24)

Cards: 9,019 (notebook 9,002); events: 240 (notebook 237). The cards table of 2026-04-23 no longer exists, so this is a reconstruction (reps = revlog rows; lapses = Again on review rows; factor = last post-answer factor).

| q | lapses | total | rate | reported lapses | reported total |
|---|---|---|---|---|---|
| Q1 (Hardest) | 208 | 2759 | 0.07539 | 205 | 2742 |
| Q2 | 15 | 2445 | 0.006135 | 15 | 2445 |
| Q3 | 13 | 2890 | 0.004498 | 13 | 2890 |
| Q4 (Easiest) | 4 | 925 | 0.004324 | 4 | 925 |

### F3. Is the lapse feeding its own predictor? (event cards only)

| quantity | value |
|---|---|
| q1_threshold | 2.65 |
| n_events | 240 |
| events_in_Q1_by_final_factor | 208 |
| events_in_Q1_by_factor_before_first_lapse | 234 |
| events_with_reviews_after_first_lapse_counted_in_duration | 240 |
| median_reps_total_for_events | 14 |
| median_reps_at_first_lapse | 5 |
| median_factor_change_at_first_lapse_pts | 200 |

### G. Source inventory (regex over the audited .py files and notebook code cells)

| file | where | kind | code |
|---|---|---|---|
| dashboard.py | line 18 | train/test or CV split | from sklearn.model_selection import KFold |
| dashboard.py | line 19 | scaler fit on full data | from sklearn.preprocessing import StandardScaler |
| dashboard.py | line 202 | scaler fit on full data | scaler = StandardScaler() |
| dashboard.py | line 203 | scaler fit on full data | X_s = scaler.fit_transform(X) |
| dashboard.py | line 206 | train/test or CV split | kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed) |
| dashboard.py | line 206 | shuffle / random state | kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed) |
| dashboard.py | line 220 | train/test or CV split | def _logistic(): return LogisticRegressionCV(cv=5, max_iter=1000, random_state=42) |
| dashboard.py | line 220 | shuffle / random state | def _logistic(): return LogisticRegressionCV(cv=5, max_iter=1000, random_state=42) |
| dashboard.py | line 221 | shuffle / random state | def _gbr():      return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42) |
| dashboard.py | line 222 | shuffle / random state | def _gbc():      return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42) |
| dashboard.py | line 505 | rolling window | daily["reviews_7d"] = daily["reviews"].rolling(7, center=True, min_periods=1).mean() |
| dashboard.py | line 506 | rolling window | daily["ret_7d"]     = daily["retention"].rolling(7, center=True, min_periods=1).mean() |
| dashboard.py | line 612 | quantile cut on full data | lift_ci        = np.quantile(lift, [0.05, 0.95]) |
| anki_causal_analysis.ipynb | cell 6, line 6 | rolling window | daily['reviews_7d'] = daily['reviews'].rolling(7, center=True).mean() |
| anki_causal_analysis.ipynb | cell 6, line 16 | rolling window | ret_7d = daily['retention'].rolling(7, center=True).mean() |
| anki_causal_analysis.ipynb | cell 10, line 10 | imputation with a full-sample statistic | ts['gap_days']  = (ts['date'] - ts['prev_date']).dt.days.fillna(1) |
| anki_causal_analysis.ipynb | cell 10, line 29 | rolling window | rollstd = ts['reviews'].rolling(14).std() |
| anki_causal_analysis.ipynb | cell 11, line 6 | rolling window | ts['ret_smooth'] = ts['retention'].rolling(7, min_periods=1, center=True).mean() |
| anki_causal_analysis.ipynb | cell 14, line 2 | scaler fit on full data | from sklearn.preprocessing import StandardScaler |
| anki_causal_analysis.ipynb | cell 14, line 26 | imputation with a full-sample statistic | psm_df['log_days_since'] = np.log1p(psm_df['days_since_last'].fillna(psm_df['days_since_last'].median())) |
| anki_causal_analysis.ipynb | cell 15, line 5 | scaler fit on full data | scaler = StandardScaler() |
| anki_causal_analysis.ipynb | cell 15, line 6 | scaler fit on full data | X_sc = scaler.fit_transform(X) |
| anki_causal_analysis.ipynb | cell 15, line 8 | shuffle / random state | lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42) |
| anki_causal_analysis.ipynb | cell 22, line 21 | quantile cut on full data | surv_fit['ef_quartile'] = pd.qcut( |
| anki_causal_analysis.ipynb | cell 25, line 5 | imputation with a full-sample statistic | hr_df['label'] = hr_df['covariate'].map({'ease_factor_c': 'Ease factor (mean-centered, +1 unit)'}).fillna(hr_d |
| psm_response_time.ipynb | cell 1, line 8 | scaler fit on full data | from sklearn.preprocessing import StandardScaler |
| psm_response_time.ipynb | cell 2, line 17 | quantile cut on full data | Q1_THRESH = reviewed['ease_factor'].quantile(0.25) |
| psm_response_time.ipynb | cell 3, line 17 | imputation with a full-sample statistic | df['log_days_since'] = np.log1p(df['days_since'].fillna(df['days_since'].median())) |
| psm_response_time.ipynb | cell 8, line 5 | scaler fit on full data | scaler = StandardScaler() |
| psm_response_time.ipynb | cell 8, line 6 | scaler fit on full data | X_sc = scaler.fit_transform(X) |
| psm_response_time.ipynb | cell 8, line 8 | shuffle / random state | lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42) |
| psm_response_time.ipynb | cell 18, line 7 | scaler fit on full data | X_fs = scaler.fit_transform(X_f) |
| psm_response_time.ipynb | cell 18, line 8 | shuffle / random state | lr_f = LogisticRegression(max_iter=1000, C=1.0, random_state=42) |
| psm_response_time_v2.ipynb | cell 1, line 8 | scaler fit on full data | from sklearn.preprocessing import StandardScaler |
| psm_response_time_v2.ipynb | cell 4, line 8 | imputation with a full-sample statistic | df['log_days_since']= np.log1p(df['days_since'].fillna(df['days_since'].median())) |
| psm_response_time_v2.ipynb | cell 6, line 4 | scaler fit on full data | scaler = StandardScaler() |
| psm_response_time_v2.ipynb | cell 6, line 5 | scaler fit on full data | X_sc = scaler.fit_transform(X) |
| psm_response_time_v2.ipynb | cell 6, line 7 | shuffle / random state | lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42) |
