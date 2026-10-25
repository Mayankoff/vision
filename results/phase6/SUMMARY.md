# Phase 6: live demo

```bash
afscreen live                                       # webcam 0 (q: quit, r: reset)
afscreen live --camera 1                            # another camera
afscreen live --video clip.avi                      # a file, played as if live
afscreen live --video clip.avi --headless --record out.mp4   # no window: status lines + annotated video
```

Code: `src/afscreen/live.py` (`LiveMonitor` holds the logic and can be tested without a camera).

## What it shows

- **Trust map** on the face: each patch from red (low trust) to green (high), from the latest window.
- **Heart rate** (last 8 s): from the spectrum, or from beats when the rhythm is irregular
  (Phase 2: the spectral peak is not valid for AF).
- **Confidence** (0-1, bar) of the latest window.
- **Rhythm** (latest 30 s, once 30 s are buffered): "regular rhythm", "possible AF - see a doctor
  for an ECG" or "signal unreliable" (beat-shape gate), from the MIMIC-trained classifier with
  isolated extra beats removed first; plus the number of **extra beats** in that window.
- **RMSSD** (latest 30 s), labelled as overestimated by the camera (Phase 2).
- **Slow / fast heart rate** only after 2 minutes of confident readings, with context
  ("can be normal for fit people...").
- **Breathing rate is not shown**: Phase 2 found it unreliable.
- "Screening aid, not a diagnosis" on screen at all times.

## How it runs

Each frame: MediaPipe landmarks and patch colours go into a rolling 32 s buffer with their real
timestamps (webcam frame rates wobble), resampled to 30 fps for analysis. Once a second the full
method runs on a snapshot in a background thread, so the video keeps its frame rate.

| step | time |
|---|---|
| per frame (landmarks + patches) | ~10 ms (Phase 5) |
| analysis of the 32 s buffer, once a second | ~220-420 ms |

POS was vectorised for this (identical output to the per-window loop, 14x faster); before, the
analysis took > 1 s and grew with the buffer.

## Checks (headless, files played as if live)

| input | truth | shown (after 35 s) |
|---|---|---|
| UBFC DATASET_1 subject 5 (real video) | oximeter HR 75-84 BPM, rising | 74-86 BPM, follows the rise; confidence 0.93-0.96; regular rhythm |
| synthetic, regular 72 BPM (60 s) | regular | 72 BPM; regular rhythm; 0 extra beats |
| synthetic, AF ~86 BPM (60 s) | AF | 83-87 BPM from beats; **possible AF** |
| synthetic, regular with extra beats (60 s) | regular + ~4 extra beats / 30 s | 70 BPM; **regular rhythm**; 4 extra beats |

The webcam itself was not tested here (the demo was not run on the user's camera); run
`afscreen live` to try it. Lighting matters: face a window or lamp, avoid strong backlight, keep
still for the first 30 s.
