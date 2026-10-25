# Phase 4: extra beats and slow / fast heart-rate flags

Commands: `afscreen extra-beats "datasets/MIT BIH" --mimic datasets/MIMIC` and `afscreen hr-flags datasets/MIMIC`.
Detailed tables: [mitbih/summary.md](mitbih/summary.md), [hr_flags/hr_flags_summary.md](hr_flags/hr_flags_summary.md),
[af_with_removal/](af_with_removal/).

## 1. Extra-beat rule (MIT-BIH Arrhythmia, beat timing only)

44 records (paced ones excluded); rule tuned on DS1, tested on DS2 (de Chazal et al., 2004); scored
outside AF / flutter. Rule: a beat whose preceding interval is > 8% shorter than the local median
and whose following interval is > 1.05 x the median.

| DS2, timing noise | sensitivity | precision | ventricular (V) found | supraventricular (A, a, J, S) found |
|---|---|---|---|---|
| 0 ms (exact) | 0.54 | 0.92 | 0.82 | 0.14 |
| **16 ms (measured on UBFC face videos)** | **0.54** | **0.86** | **0.81** | **0.15** |
| 33 ms (one frame at 30 fps) | 0.54 | 0.61 | 0.77 | 0.20 |

Ventricular extra beats are found well; atrial ones mostly not: they are often only slightly early,
so timing alone barely separates them from normal beats (a camera-only limitation; ECG methods use
the beat's shape). Refining beat times below one frame matters: at frame-level error (33 ms)
precision falls from 0.86 to 0.61.

## 2. Do extra beats cause AF false alarms, and does removing them help?

The AF classifier trained on MIMIC PPG, applied to MIT-BIH ECG beat timing (16 ms noise), share of
30 s windows called AF. The removal setting was chosen on DS1 (fewest false alarms with at most 2
points of AF sensitivity lost): remove an extra beat only if the intervals around it are regular
(within 20% of the median) and at most 20% of the window's beats are flagged.

| DS2 (test) | windows | no removal | plain removal | **chosen removal** |
|---|---|---|---|---|
| AF (sensitivity) | 170 | 0.90 | 0.71 | **0.82** |
| normal rhythm **with** extra beats (false alarms) | 443 | 0.22 | 0.09 | **0.12** |
| normal rhythm without extra beats (false alarms) | 644 | 0.003 | 0.003 | 0.005 |

- The classifier transfers to a second dataset and another sensor: AF 0.90-0.97, clean normal
  rhythm 0.2-0.3% false alarms.
- Extra beats are the main cause of false AF alarms (22% vs 0.3%), as the plan expected.
- Removal nearly halves them; the neighbour check keeps most AF (plain removal "cleans" AF too).
  The trade-off is real: 8 points of AF sensitivity on DS2. DS1 has only 59 AF windows from 2
  records, so the chosen setting is uncertain.

## 3. Effect on the full system

| | no removal | extra beats removed |
|---|---|---|
| MIMIC PERform AF, LOSO, AUC (camera-like beats) | 0.96 | 0.96 |
| MIMIC: AF subjects found / non-AF subjects correct | 19/19, 14/16 | 19/19, 14/16 |
| UBFC face videos, full method: false-alarm videos / windows | 1/12, 2/20 | **0/12, 1/20** |
| UBFC face videos, single region: false-alarm videos / windows | 2/12, 4/20 | 1/12, 2/20 |

The two MIMIC non-AF patients still flagged (012, 014) are not an extra-beat problem: they are the
two lowest-quality recordings (012: median beat-shape quality 0.70, below the 0.8 gate; 014: ECG
unusable in the heart-rate check) and their irregularity is far beyond isolated extra beats
(normalised RMSSD 0.35-0.44 vs ~0.08 for the other non-AF patients).

## 4. Slow / fast heart-rate flags (MIMIC PERform AF, 2-minute segments)

Prediction from camera-like PPG beats (30 fps, 16 ms noise, confident windows only) vs the flag
from ECG R-peaks; slow < 50, fast > 100 BPM.

| segments | scored (coverage) | agreement | Cohen's kappa | HR error |
|---|---|---|---|---|
| 350 | 333 (95%) | 0.95 | 0.88 | 2.8 BPM |

Nearly all disagreements (15 of 17) are AF patients whose ECG rate is fast but whose PPG rate reads
normal: **pulse deficit** - in AF some beats are too weak to reach the periphery, so any pulse-based
method (a camera included) undercounts the heart rate. Only 3 segments are slow, so the slow flag is
barely tested. The UBFC face videos (~1 min, 59-96 BPM) are too short for the 2-minute rule and
contain no slow or fast rates.
