# Phase 5: comparison, ablation and efficiency

Command: `afscreen compare --sizes 320x240=cache/traces_320/ubfc,cache/traces_320/ubfc1 160x120=cache/traces_160/ubfc,cache/traces_160/ubfc1 --video "datasets/UBFC/dataset 1/vid-003.avi"`.
Data: the 7 UBFC videos with a reference (heart rate, beats, RMSSD) and all 12 normal-rhythm UBFC
videos (AF false alarms); synthetic videos for motion. Raw tables in this folder.

## 1. Baselines and ablation (same downstream steps and AF classifier for every row)

AF screening uses the MIMIC-trained classifier, extra beats removed first (Phase 4) and the
beat-shape gate. Every UBFC subject has a normal rhythm, so every "AF" is a false alarm.

| variant | HR MAE (BPM) | IBI error (ms) | beat F1 | RMSSD error (ms) | AF false windows | AF false videos |
|---|---|---|---|---|---|---|
| GREEN, one region (baseline) | 0.58 | 55.4 | 0.84 | 157.0 | 14 / 17 | **11 / 12** |
| CHROM, one region (baseline) | 0.23 | 43.5 | 0.94 | 89.3 | 4 / 20 | 2 / 12 |
| POS, one region (baseline) | 0.22 | **21.1** | 0.98 | **31.4** | 2 / 20 | 1 / 12 |
| + patches, equal weights | 0.22 | 22.2 | 0.98 | 39.4 | 1 / 20 | **0 / 12** |
| + facing check | 0.23 | 21.3 | 0.98 | 41.6 | 0 / 20 | 0 / 12 |
| + brightness check | 0.23 | 21.3 | 0.98 | 41.5 | 0 / 20 | 0 / 12 |
| + consistency check | 0.23 | 21.1 | 0.98 | 38.5 | 0 / 20 | 0 / 12 |
| + beat-agreement check | 0.23 | 21.8 | 0.99 | 39.2 | 1 / 20 | 0 / 12 |
| + motion removal (full method) | 0.23 | 21.8 | 0.99 | 39.2 | 1 / 20 | 0 / 12 |

- **Heart rate alone hides the differences** (all 0.2-0.6 BPM): these videos are easy. Beat timing
  does not: it is what decides AF false alarms, and it separates the methods clearly.
- **Pulse extraction matters most:** GREEN raises a false alarm on 11 of 12 healthy people, CHROM
  on 2, POS on 1. This is why POS is the per-patch method.
- **Patches remove the remaining false alarms** (1/12 -> 0/12 videos; 2/20 -> 0-1/20 windows).
- **The individual trust checks add nothing measurable here:** on still, evenly lit faces every
  patch is good, so the weights stay nearly equal. They are designed for shadow, shine, turned
  faces and noisy patches (shown on synthetic traces in the unit tests), which UBFC does not have.
- RMSSD is slightly worse with patches than with one region (39 vs 31 ms). Both overestimate RMSSD
  (Phase 2).

## 2. Patch count vs face size (full method)

The same 7 videos re-processed at smaller frame sizes, so the face covers fewer pixels.

| frame size | skin pixels (all patches) | 6 patches | 12 | 25 | adaptive (chose) |
|---|---|---|---|---|---|
| 640x480 | 4696 | 0.23 / 21.5 | 0.23 / 20.6 | 0.22 / 20.3 | 0.23 / 21.8 (6 or 12) |
| 320x240 | 1243 | 0.23 / 22.4 | 0.22 / 22.0 | 0.21 / 22.6 | 0.23 / 22.4 (6) |
| 160x120 | 350 | 0.23 / 22.2 | 0.22 / 22.6 | 0.21 / 21.8 | 0.23 / 22.2 (6) |

HR MAE (BPM) / IBI error (ms). **Neither patch count nor face size matters on this data**, even at
~14 pixels per patch (25 patches at 160x120). The 350-pixel rule of thumb is conservative for clean,
lightly compressed, well-lit video; it is kept as a safe default, since phone video (stronger
compression, more noise) has not been tested.

## 3. Confidence measure: rhythm-independent vs spectral sharpness (from Phase 3)

MIMIC PERform AF, camera-like PPG (30 fps, 16 ms beat noise), share of 30 s windows kept:

| gate | AF windows kept | non-AF windows kept | AF classifier AUC on kept windows |
|---|---|---|---|
| none | 1.00 | 1.00 | 0.96 |
| **beat-shape consistency >= 0.8 (ours)** | **0.96** | 0.91 | **0.99** |
| spectral peak share >= 0.5 | **0.06** | 0.86 | - |
| spectral peak share >= 0.7 | 0.00 | 0.67 | - |

A spectral-sharpness confidence throws away nearly every AF window, so it would make AF
undetectable. The beat-shape gate keeps AF and normal windows alike and raises the AUC.

## 4. Motion removal

- **UBFC:** never active (no head movement passes the 1 deg / 0.5 cm threshold), so
  "+ beat-agreement" and "full" are identical.
- **Synthetic sway videos** (face moving side to side through a tinted light gradient):

| video | IBI error without / with motion removal | FFT heart rate without / with (true) |
|---|---|---|
| regular_76_sway | 2.8 / 3.5 ms | 76.0 / 76.0 (76.0) |
| af_92_sway | 4.6 / 9.1 ms | 89.1 / **74.6** (89.0) |

  POS and the patch fusion already cancel this sway; removal adds a little timing error. The 14 BPM
  miss on `af_92_sway` is the FFT heart rate, which is ill-defined for AF (three near-equal
  spectral peaks); the beat-based heart rate is right (89.2 BPM), as Phase 2 recommends.
- **Conclusion:** no benefit is shown on the available data. Without PURE (real head motion) it
  cannot be tested properly; this is a limitation, and removal stays behind its motion threshold.

## 5. Efficiency (this laptop: Intel CPU, 20 threads; 300 frames of a UBFC video)

| frame size | landmarks (MediaPipe) | patches | fusion and the rest | total | max frame rate |
|---|---|---|---|---|---|
| 640x480 | 9.0 ms | 0.7 ms | 0.7 ms | **10.4 ms** | **96 fps** |
| 320x240 | 8.5 ms | 0.5 ms | 0.4 ms | 9.4 ms | 107 fps |

About three times faster than real time at 30 fps; MediaPipe takes 87% of the time, so a live
webcam demo (Phase 6) is feasible.

## Not done

- Pretrained deep baselines (PhysNet, TS-CAN, EfficientPhys, PhysFormer via rPPG-Toolbox): the
  toolbox setup was removed with the earlier project and was not rebuilt.
- Small test set: 7 videos with a reference, 12 for false alarms. Differences of one video or
  one window are not statistically meaningful; the GREEN / CHROM / POS gap is the robust result.
