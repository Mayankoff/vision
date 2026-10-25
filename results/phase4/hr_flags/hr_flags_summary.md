# Phase 4: slow / fast heart-rate flags on MIMIC PERform AF

2-minute segments; reference = ECG R-peaks (XQRS); prediction = PPG resampled to 30 fps with 16 ms beat-timing noise, confident 30 s windows only (beat shape >= 0.8). Thresholds: slow < 50, fast > 100 BPM.

| segments | scored | coverage | accuracy | kappa | hr_mae_bpm |
|---|---|---|---|---|---|
| 350.000 | 333.000 | 0.951 | 0.949 | 0.881 | 2.820 |

Confusion (rows = reference, columns = prediction):

| ref_flag | slow | normal | fast |
|---|---|---|---|
| slow | 3 | 0 | 0 |
| normal | 0 | 223 | 2 |
| fast | 0 | 15 | 90 |

