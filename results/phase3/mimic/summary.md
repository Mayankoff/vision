# Phase 3: AF classifier on MIMIC PERform AF

35 subjects (19 AF, 16 non-AF); 30 s windows; leave-one-subject-out cross-validation. `subj_*` = per subject (mean window probability).

## contact_125Hz: 1400 windows (760 AF)

Beat-shape gating (rhythm-independent; the proposed confidence part):

| threshold | windows | coverage_af | coverage_non_af | subjects | logreg_auc | logreg_sens | logreg_spec | logreg_subj_sens | logreg_subj_spec | forest_auc | forest_sens | forest_spec | forest_subj_sens | forest_subj_spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1400.00 | 1.00 | 1.00 | 35.00 | 0.85 | 0.89 | 0.76 | 0.89 | 0.88 | 0.94 | 0.95 | 0.86 | 1.00 | 0.88 |
| 0.50 | 1400.00 | 1.00 | 1.00 | 35.00 | 0.85 | 0.89 | 0.76 | 0.89 | 0.88 | 0.94 | 0.95 | 0.86 | 1.00 | 0.88 |
| 0.60 | 1400.00 | 1.00 | 1.00 | 35.00 | 0.85 | 0.89 | 0.76 | 0.89 | 0.88 | 0.94 | 0.95 | 0.86 | 1.00 | 0.88 |
| 0.70 | 1400.00 | 1.00 | 1.00 | 35.00 | 0.85 | 0.89 | 0.76 | 0.89 | 0.88 | 0.94 | 0.95 | 0.86 | 1.00 | 0.88 |
| 0.80 | 1386.00 | 1.00 | 0.98 | 35.00 | 0.87 | 0.91 | 0.79 | 0.95 | 0.88 | 0.94 | 0.94 | 0.87 | 1.00 | 0.88 |
| 0.90 | 1026.00 | 0.73 | 0.74 | 34.00 | 0.96 | 0.97 | 0.91 | 1.00 | 0.93 | 0.97 | 0.95 | 0.95 | 1.00 | 0.93 |

Spectral peak-share gating (assumes a regular rhythm; for comparison): coverage only

| threshold | windows | coverage_af | coverage_non_af |
|---|---|---|---|
| 0.00 | 1400.00 | 1.00 | 1.00 |
| 0.50 | 446.00 | 0.01 | 0.68 |
| 0.60 | 283.00 | 0.00 | 0.44 |
| 0.70 | 166.00 | 0.00 | 0.26 |
| 0.80 | 110.00 | 0.00 | 0.17 |
| 0.90 | 59.00 | 0.00 | 0.09 |

## camera_30Hz: 1399 windows (759 AF)

Beat-shape gating (rhythm-independent; the proposed confidence part):

| threshold | windows | coverage_af | coverage_non_af | subjects | logreg_auc | logreg_sens | logreg_spec | logreg_subj_sens | logreg_subj_spec | forest_auc | forest_sens | forest_spec | forest_subj_sens | forest_subj_spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1399.00 | 1.00 | 1.00 | 35.00 | 0.91 | 0.91 | 0.85 | 1.00 | 0.88 | 0.96 | 0.94 | 0.88 | 1.00 | 0.88 |
| 0.50 | 1399.00 | 1.00 | 1.00 | 35.00 | 0.91 | 0.91 | 0.85 | 1.00 | 0.88 | 0.96 | 0.94 | 0.88 | 1.00 | 0.88 |
| 0.60 | 1399.00 | 1.00 | 1.00 | 35.00 | 0.91 | 0.91 | 0.85 | 1.00 | 0.88 | 0.96 | 0.94 | 0.88 | 1.00 | 0.88 |
| 0.70 | 1372.00 | 0.99 | 0.96 | 35.00 | 0.93 | 0.92 | 0.88 | 1.00 | 0.88 | 0.98 | 0.95 | 0.90 | 1.00 | 0.88 |
| 0.80 | 1307.00 | 0.96 | 0.91 | 35.00 | 0.96 | 0.94 | 0.92 | 1.00 | 0.88 | 0.99 | 0.97 | 0.95 | 1.00 | 0.88 |
| 0.90 | 1121.00 | 0.76 | 0.85 | 34.00 | 0.99 | 0.97 | 0.97 | 1.00 | 0.93 | 1.00 | 0.98 | 0.99 | 1.00 | 0.93 |

Spectral peak-share gating (assumes a regular rhythm; for comparison): coverage only

| threshold | windows | coverage_af | coverage_non_af |
|---|---|---|---|
| 0.00 | 1399.00 | 1.00 | 1.00 |
| 0.50 | 592.00 | 0.06 | 0.86 |
| 0.60 | 535.00 | 0.01 | 0.82 |
| 0.70 | 429.00 | 0.00 | 0.67 |
| 0.80 | 242.00 | 0.00 | 0.38 |
| 0.90 | 70.00 | 0.00 | 0.11 |

## camera_30Hz_jitter16ms: 1399 windows (759 AF)

Beat-shape gating (rhythm-independent; the proposed confidence part):

| threshold | windows | coverage_af | coverage_non_af | subjects | logreg_auc | logreg_sens | logreg_spec | logreg_subj_sens | logreg_subj_spec | forest_auc | forest_sens | forest_spec | forest_subj_sens | forest_subj_spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1399.00 | 1.00 | 1.00 | 35.00 | 0.91 | 0.87 | 0.83 | 0.95 | 0.88 | 0.96 | 0.93 | 0.89 | 1.00 | 0.88 |
| 0.50 | 1399.00 | 1.00 | 1.00 | 35.00 | 0.91 | 0.87 | 0.83 | 0.95 | 0.88 | 0.96 | 0.93 | 0.89 | 1.00 | 0.88 |
| 0.60 | 1399.00 | 1.00 | 1.00 | 35.00 | 0.91 | 0.87 | 0.83 | 0.95 | 0.88 | 0.96 | 0.93 | 0.89 | 1.00 | 0.88 |
| 0.70 | 1372.00 | 0.99 | 0.96 | 35.00 | 0.93 | 0.89 | 0.86 | 0.95 | 0.88 | 0.97 | 0.94 | 0.90 | 1.00 | 0.88 |
| 0.80 | 1307.00 | 0.96 | 0.91 | 35.00 | 0.96 | 0.93 | 0.90 | 0.95 | 0.88 | 0.99 | 0.96 | 0.95 | 1.00 | 0.88 |
| 0.90 | 1121.00 | 0.76 | 0.85 | 34.00 | 0.99 | 0.96 | 0.95 | 1.00 | 0.93 | 0.99 | 0.97 | 0.97 | 1.00 | 0.93 |

Spectral peak-share gating (assumes a regular rhythm; for comparison): coverage only

| threshold | windows | coverage_af | coverage_non_af |
|---|---|---|---|
| 0.00 | 1399.00 | 1.00 | 1.00 |
| 0.50 | 592.00 | 0.06 | 0.86 |
| 0.60 | 535.00 | 0.01 | 0.82 |
| 0.70 | 429.00 | 0.00 | 0.67 |
| 0.80 | 242.00 | 0.00 | 0.38 |
| 0.90 | 70.00 | 0.00 | 0.11 |

