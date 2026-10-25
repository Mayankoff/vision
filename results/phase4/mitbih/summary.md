# Phase 4: extra-beat detection on MIT-BIH Arrhythmia

44 records (paced 102, 104, 107, 217 excluded). Rule tuned on DS1 (best F1 at 16 ms timing noise), tested on DS2; scored outside AF / flutter; fusion and unclassifiable beats ignored.

**Rule:** IBI before < (1 - 0.08) x local median, IBI after > 1.05 x local median (median of 8 intervals each side).

## DS2 test

| jitter_ms | sensitivity | precision | F1 | sveb_sens | veb_sens | false_per_1000_normal |
|---|---|---|---|---|---|---|
| 0.000 | 0.540 | 0.924 | 0.682 | 0.136 | 0.816 | 5.479 |
| 16.000 | 0.539 | 0.862 | 0.663 | 0.148 | 0.806 | 10.653 |
| 33.000 | 0.540 | 0.608 | 0.572 | 0.202 | 0.771 | 43.082 |

## Top of the DS1 grid

| prematurity | recovery | sensitivity | precision | F1 | sveb_sens | veb_sens | false_per_1000_normal |
|---|---|---|---|---|---|---|---|
| 0.080 | 1.050 | 0.639 | 0.910 | 0.751 | 0.416 | 0.702 | 6.345 |
| 0.100 | 1.050 | 0.614 | 0.951 | 0.746 | 0.406 | 0.672 | 3.126 |
| 0.040 | 1.100 | 0.618 | 0.938 | 0.745 | 0.314 | 0.704 | 4.066 |
| 0.100 | 1.000 | 0.638 | 0.892 | 0.744 | 0.466 | 0.687 | 7.709 |
| 0.060 | 1.100 | 0.604 | 0.957 | 0.740 | 0.305 | 0.687 | 2.726 |

## Share of beats flagged per 30 s window (removal is skipped above 0.20)

| index | count | mean | std | min | 10% | 50% | 90% | max | above_max_share |
|---|---|---|---|---|---|---|---|---|---|
| AF | 229.000 | 0.133 | 0.060 | 0.000 | 0.047 | 0.136 | 0.207 | 0.275 | 0.122 |
| non-AF with extra beats | 1046.000 | 0.116 | 0.109 | 0.000 | 0.019 | 0.072 | 0.286 | 0.488 | 0.228 |
| non-AF without | 1274.000 | 0.007 | 0.018 | 0.000 | 0.000 | 0.000 | 0.031 | 0.160 | 0.000 |

## AF classifier (trained on MIMIC) on MIT-BIH 30 s windows: share called AF

Beat times from the ECG annotations with 16 ms timing noise. `none` = no removal; `capX` = extra beats removed unless more than X of the beats are flagged; `_tolY` = only extra beats whose neighbouring intervals are within Y of the local median. For AF windows this is sensitivity; for the others, the false-alarm rate.

### DS1

| group | windows | none | cap0.2 | cap0.2_tol0.08 | cap0.2_tol0.1 | cap0.2_tol0.15 | cap0.2_tol0.2 | cap1 | cap1_tol0.08 | cap1_tol0.1 | cap1_tol0.15 | cap1_tol0.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AF | 59 | 0.966 | 0.915 | 0.983 | 1.000 | 1.000 | 0.983 | 0.881 | 0.983 | 1.000 | 1.000 | 0.966 |
| non-AF with extra beats | 602 | 0.131 | 0.055 | 0.095 | 0.088 | 0.076 | 0.068 | 0.055 | 0.101 | 0.093 | 0.078 | 0.068 |
| non-AF without | 630 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 |

### DS2

| group | windows | none | cap0.2 | cap0.2_tol0.08 | cap0.2_tol0.1 | cap0.2_tol0.15 | cap0.2_tol0.2 | cap1 | cap1_tol0.08 | cap1_tol0.1 | cap1_tol0.15 | cap1_tol0.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AF | 170 | 0.900 | 0.712 | 0.888 | 0.882 | 0.859 | 0.824 | 0.682 | 0.894 | 0.888 | 0.859 | 0.812 |
| non-AF with extra beats | 443 | 0.221 | 0.088 | 0.169 | 0.147 | 0.122 | 0.122 | 0.054 | 0.160 | 0.135 | 0.102 | 0.108 |
| non-AF without | 644 | 0.003 | 0.003 | 0.003 | 0.003 | 0.005 | 0.005 | 0.003 | 0.003 | 0.003 | 0.005 | 0.005 |

**Chosen on DS1** (fewest false alarms with at most 2% AF sensitivity lost): `cap0.2_tol0.2`. DS2: AF 0.90 -> 0.82; non-AF with extra beats 0.22 -> 0.12.

