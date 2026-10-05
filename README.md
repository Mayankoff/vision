# Contactless Heart-Rate Estimation from Facial Video (rPPG)

BCSE417L Machine Vision course project. Mayank (23BAI0158), Vyomesh Pandita (23BAI0151).

The pipeline estimates heart rate (BPM) from an ordinary RGB facial video using remote
photoplethysmography: MediaPipe face landmarks → forehead/cheek skin ROIs → RGB traces →
classical rPPG methods (GREEN, ICA, CHROM, POS) → pulse waveform → heart rate. Deep models
(PhysNet, TS-CAN, EfficientPhys, PhysFormer) are run through the
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) in Phase 3.

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the phased plan.

## Status

| Phase | State |
|---|---|
| 0 - Setup, protocol, splits | Done in code. **Still to do by the team:** request PURE + MMPD access, download UBFC-rPPG, set up the toolbox environment, run `make_splits.py --check` on each dataset. |
| 1 - Own classical pipeline | Implemented and tested on synthetic data. **Next:** run on UBFC-rPPG. |
| 2 - Toolbox baselines, cross-check, ROI ablation | Implemented and tested end to end (both environments, Windows scripts) on synthetic data. **Next:** run on the real datasets. |
| 3-7 | Not started. |

## Setup (Windows)

Everything below is PowerShell, run from the repo root. Tested scripts are in `scripts\windows\`.
Clone to a **short path** (e.g. `C:\vision`): Windows limits paths to 260 characters by default
and the toolbox's cache paths are long.

```powershell
git clone --recurse-submodules https://github.com/mayankoff/vision.git C:\vision
cd C:\vision
```

### 1. Our pipeline - Python 3.11, 3.12 or 3.13 (not 3.10), CPU is enough

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
.venv\Scripts\Activate.ps1
```

This creates `.venv`, installs the pinned requirements and the `rppg` package, downloads the
MediaPipe face model (3.7 MB, to `models\`) and runs the tests.

### 2. rPPG-Toolbox environment - Python 3.8, no conda needed

The toolbox needs Python 3.8 + PyTorch 2.1.2, which current MediaPipe does not support, so it
gets its **own** environment, `.venv-toolbox`; the two halves exchange files only. The script uses
`uv` (installed into `.venv`), which downloads Python 3.8 by itself:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\setup_toolbox_env.ps1        # NVIDIA GPU (CUDA 12.1)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\setup_toolbox_env.ps1 -Cpu   # no NVIDIA GPU: enough for Phase 2
```

The phase scripts find `.venv-toolbox` automatically. (If you prefer conda, add `-Conda`.)

The toolbox's `setup.sh` is bash-only and builds `mamba-ssm`, which only compiles on Linux +
CUDA. The script skips it; `scripts\toolbox\stubs\mamba_ssm` stands in so the toolbox still
imports (only PhysMamba, which is not in our plan, is unusable). The toolbox is pinned at commit
`b7500b848f84ad7f86e277b4612563b69f4f88f9`.

Never run the toolbox's `main.py` directly; use `scripts\toolbox\run_toolbox.py` (or the
phase scripts), which handles the stub, Windows multiprocessing and repo-relative config paths.

<details><summary>Linux / macOS</summary>

```bash
sudo apt-get install -y libegl1 libgles2      # Linux only: MediaPipe needs EGL / GLES
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cd external/rPPG-Toolbox && bash setup.sh conda   # toolbox env (Linux builds mamba-ssm too)
```
Then use the Python commands under "Usage" directly.
</details>

## Data

Raw datasets go under `data/` (gitignored), in the same layout rPPG-Toolbox expects, so one
copy serves both codebases:

```
data\UBFC-rPPG\subject1\{vid.avi, ground_truth.txt} ...         # DATASET_2, open download
data\PURE\01-01\{01-01\Image*.png, 01-01.json} ...              # request form, TU Ilmenau
data\MMPD\subject1\{p1_0.mat, p1_1.mat, ...} ...                 # licence agreement; ask for mini-MMPD
```

| Dataset | Access |
|---|---|
| UBFC-rPPG | https://sites.google.com/view/ybenezeth/ubfcrppg |
| PURE | https://www.tu-ilmenau.de/en/university/faculties/faculty-of-computer-science-and-automation/profile/institutes-and-facilities/institute-for-technical-informatics-and-automation/data-sets-code/pulse-rate-detection-dataset-pure |
| MMPD | https://github.com/THU-CS-PI/MMPD_rPPG_dataset |

## Usage

### Quickest real test: your own face

Record ~30 s of your face with a phone or webcam (good light, sit still), ideally while wearing a
smartwatch or pulse oximeter to compare against, then:

```powershell
rppg my_face.mp4 --plot pulse.png --debug-video rois.mp4
python scripts\figure_pipeline.py my_face.mp4 --out fig_pipeline.png    # report figure, stage by stage
```

### Phase 1 - our classical pipeline on a dataset

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run_phase1.ps1 -Dataset ubfc   # add -DebugVideos for ROI overlays
```

which runs:

```powershell
# 1. face landmarks + ROI averaging (slow, ~10-30 ms/frame per worker; resumable; once per dataset)
python scripts\extract_traces.py ubfc data\UBFC-rPPG --out cache\traces\ubfc --workers 4
# 2. every classical method x every ROI, scored with the frozen protocol (seconds)
python scripts\run_classical.py cache\traces\ubfc --out results\classical\ubfc --window 10
# 3. the same on the test subjects only (to compare with the deep models in Phase 3)
python scripts\run_classical.py cache\traces\ubfc --split splits\ubfc.json --subset test --out results\classical\ubfc_test
```

### Phase 2 - toolbox baselines, cross-check, ROI ablation

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run_phase2.ps1                    # every dataset present in data\
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run_phase2.ps1 -Datasets ubfc     # just one
```

For each dataset it runs Phase 1 if needed, then:

```powershell
# toolbox environment: toolbox preprocessing (Haar-cascade face crop) + its 7 classical methods, per-video outputs
.venv-toolbox\Scripts\python.exe scripts\toolbox\dump_unsupervised.py --config_file configs\toolbox\UBFC-rPPG_UNSUPERVISED.yaml
# our environment: re-score them with OUR protocol, same CSV format as run_classical.py
python scripts\score_predictions.py cache\toolbox\predictions\UBFC-rPPG --out results\toolbox\ubfc
# tables + figures
python scripts\phase2_report.py results\classical\ubfc_per_video.csv results\toolbox\ubfc_per_video.csv --out results\phase2
```

`results\phase2\REPORT.md` contains: (1) all methods - ours on the skin ROI and on a face box,
and the toolbox's 7 methods - on every dataset; (2) a video-by-video cross-check of our GREEN /
ICA / CHROM / POS against the toolbox's (median HR difference, % within 3 BPM, paired Wilcoxon
test on absolute errors); (3) the ROI ablation - forehead, each cheek, all skin, face box - with
paired tests against the skin ROI.

Notes:
- The toolbox's stock MMPD config evaluates only a subset (stationary, skin type 3, no exercise,
  no natural light); the toolbox paper's MMPD numbers are on that subset. Our config uses **all**
  MMPD videos, for the Phase 4 condition analysis.
- The toolbox names MMPD outputs by subject + condition labels (not video index), so MMPD
  videos are matched between the codebases on those labels.
- The first toolbox run per dataset caches face crops in `cache\toolbox\preprocessed`; later runs reuse them.

### Splits

```powershell
python scripts\make_splits.py --check ubfc data\UBFC-rPPG      # verify against the data on disk (do this first)
```

Subject-independent, ~70/15/15, seed 2025: UBFC 30/6/6, PURE 6/2/2, MMPD 23/5/5 subjects.
Do not regenerate after experiments start.

### Smoke test without real data

Builds a UBFC-format dataset from one face photo with known heart rates; `--hard` adds lighting
flicker, a weaker pulse and more head motion so the methods separate.

```powershell
python scripts\make_synthetic_dataset.py face.jpg --out cache\synthetic --hard --duration 30 --hrs 52,61,67,74,80,88,95,103,118,135
python scripts\extract_traces.py ubfc cache\synthetic --out cache\traces\synthetic --workers 4
python scripts\run_classical.py cache\traces\synthetic --out cache\results\synthetic
```

## Evaluation protocol

Frozen in [`src/rppg/protocol.py`](src/rppg/protocol.py) and applied identically to every
method:

| Item | Setting |
|---|---|
| HR band | 0.7-3.0 Hz (42-180 BPM) |
| Post-processing (all methods) | smoothness-priors detrend (λ=100) → Butterworth band-pass (zero-phase) → standardise |
| HR estimator | FFT periodogram peak (primary); peak detection, 60 / median IBI (secondary) |
| Evaluation window | whole video (primary); `--window 10` for per-10 s results |
| Ground-truth HR | from the reference BVP with the same processing and estimator, not the oximeter's displayed HR |
| Metrics | MAE, RMSE, MAPE, Pearson r over per-video HR; SNR (dB) averaged over videos |
| SNR | power within ±6 BPM of GT HR and its 1st harmonic vs. rest of band (as rPPG-Toolbox) |

The toolbox's own post-processing hard-codes a 0.6-3.3 Hz band, so toolbox outputs are
re-scored with `rppg.protocol` (`scripts/score_predictions.py`) rather than used as-is.

## Tests

```powershell
pytest                                              # 67 fast tests, ~3 s
$env:RPPG_FACE_IMAGE="face.jpg"; pytest -m slow     # end-to-end through MediaPipe on a synthetic pulsing face
```

The tests check each method recovers known heart rates (45-170 BPM) from synthetic RGB traces,
that CHROM/POS cancel a brightness flicker that fools GREEN, the dataset loaders on miniature
fake datasets in each format, face-dropout handling, and the metrics against hand-computed values.

## Repository layout

```
src/rppg/
  protocol.py          frozen evaluation settings
  io/datasets.py       Block 1  - UBFC-rPPG / PURE / MMPD readers, BVP alignment
  face/landmarker.py   Block 2  - MediaPipe FaceLandmarker wrapper
  face/roi.py          Block 3  - forehead / cheek polygons, face box
  signal/preprocess.py Block 4  - spatial mean, gap filling, resampling, detrend, band-pass
  signal/methods.py    Block 5A - GREEN, ICA, CHROM, POS
  signal/hr.py         Block 7  - FFT and peak-detection HR
  eval/metrics.py      Block 9  - MAE, RMSE, MAPE, Pearson, SNR
  eval/plotstyle.py    shared figure style (colour-blind-safe palette)
  pipeline.py          trace extraction (cached) + estimation
  synth.py             synthetic traces / videos with known HR
  cli.py               `rppg` command
scripts/               make_splits, extract_traces, run_classical, score_predictions, phase2_report,
                       figure_pipeline, make_synthetic_dataset
scripts/toolbox/       run in the TOOLBOX env: dump_unsupervised, run_toolbox (main.py wrapper), mamba stub
scripts/windows/       PowerShell: setup, setup_toolbox_env, run_phase1, run_phase2
configs/toolbox/       our toolbox configs (repo-relative paths)
splits/                fixed subject splits (committed)
external/rPPG-Toolbox  pinned submodule
```

## Implementation notes

- **MediaPipe API.** The legacy `mp.solutions.face_mesh` API is no longer shipped in current
  mediapipe; we use the Tasks `FaceLandmarker` (478 landmarks = 468 Face Mesh + 10 iris, so
  Face Mesh indices still apply). It tracks across frames in video mode. If the face is lost,
  the last landmarks are reused for up to 0.5 s; longer gaps are interpolated and reported as
  `face_detection_rate`.
- **CHROM** band-passes the chrominance signals over the whole recording, then computes alpha
  per 1.6 s window. rPPG-Toolbox filters inside each 1.6 s window, which on synthetic tests
  let the 2nd harmonic win below ~60 BPM.
- **ICA** uses scikit-learn's FastICA (the toolbox uses JADE) and picks the component with the
  most dominant spectral peak inside the HR band.
- **Peak-detection HR** suppresses peaks closer than 60 % of the dominant period (avoids
  counting the dicrotic notch as a beat), refines peaks to sub-sample precision, and uses the
  median IBI.
