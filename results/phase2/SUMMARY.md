# Phase 2: breathing, heart rate, beat timing and HRV

Code: `src/afscreen/signal/resp.py`, `signal/hrv.py`, `signal/spectrum.py` (breathing-aware HR),
`signal/beats.py` (sub-frame refinement), `vitals.py` (experiments). Raw tables: `breathing_*.csv`,
`hr_*.csv`, `hrv_*.csv`.

Data: MIMIC PERform Testing adults (100 ICU patients, 10 min) and MIMIC PERform AF recordings with
respiration (26), with the PPG resampled to 30 fps as a camera stand-in; references are impedance
pneumography and ECG R-peaks. HRV uses the 7 UBFC face videos with a reference. COHFACE (the plan's
breathing set) was not available.

## 1. Breathing rate - not reliable enough to be an output

Karlen et al. (2013) modulations (amplitude RIAV, interval RIFV, intensity RIIV) with smart fusion.
Best setting: 64 s windows, wide-band pulse (0.1-3.3 Hz; keeps RIIV, which the POS pulse loses),
all three modulations, fusion SD <= 4 breaths/min. "Constant guess" = always the median reference
rate (17.9 /min).

| group | windows | coverage | MAE (/min) | within 2 /min | constant guess: MAE / within 2 |
|---|---|---|---|---|---|
| ICU non-AF | 179 | 0.80 | 3.8 | 0.43 | 3.3 / 0.21 |
| ICU adults (Testing) | 573 | 0.79 | 5.5 | 0.39 | 4.1 / 0.26 |
| ICU AF | 142 | 0.82 | 9.8 | 0.07 | 5.7 / 0.24 |

- The estimate is right about twice as often as guessing (within 2 /min), but its misses are large,
  so its mean error is **worse than a constant guess**. A stricter fusion threshold helps a little
  (non-AF: MAE 3.1 at SD <= 1, but only 31% coverage).
- In AF it fails: the irregular beats swamp the breathing modulation (and RIFV carries none).
- Choices that helped: the wide-band pulse (non-AF MAE 5.4 -> 3.8 with RIIV) and 64 s rather
  than 32 s windows.
- ICU data is a hard case (ventilated, ill patients); a resting healthy set such as COHFACE could
  look better, but we cannot show it. **Decision:** breathing is not shown as a vital sign; it is
  kept only as an experimental helper.

## 2. Heart rate per 30 s window vs ECG

| group | windows | irregular | FFT peak | breathing-aware FFT | beats (60 / mean IBI) | combined |
|---|---|---|---|---|---|---|
| ICU AF | 760 | 100% | 12.5 (35% > 10) | 12.0 (35%) | **7.4 (19%)** | **7.4 (19%)** |
| ICU non-AF | 602 | 30% | **2.2 (6%)** | 2.4 (6%) | 2.6 (6%) | 2.6 (6%) |
| ICU adults (Testing) | 1989 | 55% | **8.5 (15%)** | 8.5 (16%) | 8.8 (16%) | 9.0 (17%) |

MAE in BPM (share of windows off by more than 10 BPM). Combined = beats when the window is
irregular (normalised RMSSD > 0.1), else breathing-aware FFT.

- **Beat-based HR is the right estimate in AF:** the FFT peak is unreliable for an irregular
  rhythm (no single frequency); beats halve the large errors (35% -> 19%).
- **Breathing-aware HR does not help:** where it changed the estimate by more than 3 BPM (484
  windows) it was better in 241 and worse in 243 - a coin flip, as expected from the unreliable
  breathing rate. Kept as a function, not used by default.
- The Testing adults are noisier than the AF study's recordings (55% of windows look irregular),
  so their errors are higher for every method.

## 3. Beat timing and HRV on the UBFC face videos (7 videos, ~1 min each)

| variant | beats placed | IBI error (ms) | RMSSD error (ms) | SDNN error (ms) | RMSSD: camera vs reference |
|---|---|---|---|---|---|
| single region | whole frames | 25.4 | 36.6 | 18.6 | 75.6 vs 39.0 |
| single region | **parabolic** | **21.0** | **31.3** | **16.8** | 70.3 vs 39.0 |
| single region | cubic spline | 21.1 | 31.6 | 16.9 | 70.6 vs 39.0 |
| full method | parabolic | 21.6 | 39.1 | 18.5 | 78.1 vs 39.0 |

- Sub-frame refinement matters (IBI error 25 -> 21 ms); a cubic spline is no better than a parabola.
- **The camera roughly doubles RMSSD** (70-78 vs 39 ms): ~20 ms of beat-timing noise adds
  beat-to-beat variation that is not there. The plan's comparison point (RMSSD error 10.5 ms,
  SDNN 6.15 ms [15]) is **not reached** (31 and 17 ms). This is the same noise that caused the
  AF false alarms (Phase 3), which the classifier was trained to tolerate.
- Combining beat times across patches (trust-weighted, per-patch delay removed) did not reduce
  the error (24.1 vs 24.2 ms, prototype): the timing noise is shared by all patches (lighting,
  camera, or real face-vs-finger differences), so it is not averaged away. Not kept.
- The full method is slightly worse than the single region for HRV (window-by-window weight
  changes may shift beat times slightly); for heart rate they tie (Phase 1).
