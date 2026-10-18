### (a) Real estimate against the shuffled-label null

| analysis | statistic | real estimate | null sd | null 2.5% | null 97.5% | perm p (unrestricted) | perm p (within month) | B |
|---|---|---|---|---|---|---|---|---|
| PSM v2 | median_diff | 1084 | 399.7 | -834.2 | 757.6 | 0.01198 | 0.007984 | 500 |
| PSM Phase 3 | mean_diff | -0.0007273 | 0.002596 | -0.005091 | 0.005473 | 0.8942 | 0.8842 | 500 |
| DML ridge | theta (log RT) | 0.138 | 0.03491 | -0.07187 | 0.07088 | 0.001996 | 0.001996 | 500 |
| DML gbm | theta (log RT) | 0.1634 | 0.03391 | -0.06693 | 0.06381 | 0.001996 | 0.001996 | 500 |

### (b) and (c) Placebo outcomes

| analysis | outcome | n | estimate | null mean | null sd | centred z | perm p | perm p (centred) | analytic p | cluster-by-card p |
|---|---|---|---|---|---|---|---|---|---|---|
| PSM v2 | (b) previous review's response time | 277 | -302 | 34.78 | 419.2 | -0.8034 | 0.4691 | 0.4251 | 0.8395 | nan |
| PSM v2 | (c) kanji count of card | 308 | 0.05195 | 0.1789 | 0.1757 | -0.7227 | 0.8663 | 0.4551 | 0.9207 | nan |
| PSM v2 | (c) note position in deck | 308 | 0.0329 | 0.05718 | 0.02379 | -1.021 | 0.8383 | 0.3154 | 0.155 | nan |
| PSM Phase 3 | (b) previous review retained | 1149 | 0.004352 | -7.912e-05 | 0.003375 | 1.313 | 0.2196 | 0.1896 | nan | nan |
| PSM Phase 3 | (c) kanji count of card | 1375 | 0.52 | 0.4893 | 0.07312 | 0.4194 | 0.3353 | 0.6527 | 8.215e-23 | nan |
| PSM Phase 3 | (c) note position in deck | 1375 | 0.1405 | 0.1547 | 0.01228 | -1.156 | 0.8802 | 0.2495 | 4.416e-38 | nan |
| DML ridge | (b) previous review's log response time | 4532 | 0.02245 | -0.004045 | 0.03873 | 0.6842 | 0.6028 | 0.5269 | 0.5729 | 0.574 |
| DML ridge | (c) kanji count of card | 4736 | 0.07975 | -0.001327 | 0.09262 | 0.8754 | 0.3832 | 0.3653 | 0.4213 | 0.4441 |
| DML ridge | (c) note position in deck | 4736 | 0.002665 | 0.0004489 | 0.01371 | 0.1616 | 0.8403 | 0.8623 | 0.8384 | 0.8433 |
| DML gbm | (b) previous review's log response time | 4532 | 0.04023 | -0.003827 | 0.03878 | 1.136 | 0.3253 | 0.2675 | 0.3265 | 0.3204 |
| DML gbm | (c) kanji count of card | 4736 | 0.0978 | 0.001347 | 0.09232 | 1.045 | 0.2814 | 0.2894 | 0.3367 | 0.3356 |
| DML gbm | (c) note position in deck | 4736 | 0.004026 | 0.0002221 | 0.01368 | 0.2781 | 0.7705 | 0.7784 | 0.7679 | 0.7629 |

### Matching diagnostics (real labels)

| analysis | n_rows | n_treated | n_controls | n_pairs | n_unique_controls_used | max_times_one_control_used | share_of_pairs_on_10_most_reused_controls | median_position_of_matched_controls_in_row_order | spearman_control_row_position_vs_times_used | share_of_rows_sharing_a_ps_value_with_another_row | frame_sorted_by_cid | share_rows_with_missing_card | mean_deck_position_all_treated | mean_deck_position_matched_treated | mean_deck_position_matched_controls | mean_deck_position_all_controls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PSM v2 | 4038 | 309 | 3729 | 309 | 262 | 5 | 0.1133 | 0.4486 | -0.02118 | 0.2907 | True | 0.004458 | 0.5702 | 0.5702 | 0.538 | 0.5647 |
| PSM Phase 3 | 21803 | 1375 | 20428 | 1375 | 952 | 54 | 0.2073 | 0.2762 | -0.03847 | 0.3541 | True | 0.001238 | 0.499 | 0.499 | 0.3586 | 0.5171 |

### Row-order sensitivity of the PSM estimate

| analysis | estimate | audited order | across row orders: mean | sd | min | max | share of orders with p<0.05 | same sign as audited | B |
|---|---|---|---|---|---|---|---|---|---|
| PSM v2 | response time, ms (median_diff) | 1084 | 1028 | 158.1 | 432 | 1698 | 0.936 | 1 | 500 |
| PSM Phase 3 | retention (0/1) (mean_diff) | -0.0007273 | -0.0007273 | 0 | -0.0007273 | -0.0007273 | 0 | 1 | 500 |
