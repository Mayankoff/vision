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
| 0 - Setup, protocol, splits | Done in code. **Still to do by the team:** request PURE + MMPD access, download UBFC-rPPG, set up the toolbox environment on a GPU machine, run `make_splits.py --check` on each dataset. |
| 1 - Own classical pipeline | Implemented and tested on synthetic data. **Next:** run it on UBFC-rPPG (commands below). |
| 2-7 | Not started. |

## Setup

### 1. Our pipeline (Python ≥ 3.10, CPU is enough)

```bash
# MediaPipe needs OpenGL ES / EGL libraries on Linux (preinstalled on most desktops):
sudo apt-get install -y libegl1 libgles2

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The MediaPipe face-landmarker model (`models/face_landmarker.task`, 3.7 MB) is downloaded
automatically on first use.

### 2. rPPG-Toolbox (separate environment, Phase 3)

The toolbox is a git submodule pinned at commit `b7500b848f84ad7f86e277b4612563b69f4f88f9`.
It needs Python 3.8 + CUDA (PyTorch 2.1.2, mamba-ssm), which is incompatible with current
MediaPipe, so it lives in **its own environment**. The two halves exchange files only.

```bash
git submodule update --init
cd external/rPPG-Toolbox
bash setup.sh conda        # or: bash setup.sh uv
```

## Data

Raw datasets go under `data/` (gitignored), in the same layout rPPG-Toolbox expects, so one
copy serves both codebases:

```
data/UBFC-rPPG/subject1/{vid.avi, ground_truth.txt} ...          # DATASET_2, open download
data/PURE/01-01/{01-01/Image*.png, 01-01.json} ...                # request form, TU Ilmenau
data/MMPD/subject1/{p1_0.mat, p1_1.mat, ...} ...                  # licence agreement; ask for mini-MMPD
```

| Dataset | Access |
|---|---|
| UBFC-rPPG | https://sites.google.com/view/ybenezeth/ubfcrppg |
| PURE | https://www.tu-ilmenau.de/en/university/faculties/faculty-of-computer-science-and-automation/profile/institutes-and-facilities/institute-for-technical-informatics-and-automation/data-sets-code/pulse-rate-detection-dataset-pure |
| MMPD | https://github.com/THU-CS-PI/MMPD_rPPG_dataset |

## Usage

### Single video (offline prototype)

```bash
rppg path/to/video.avi                                   # POS on forehead + cheeks
rppg video.avi --method chrom --plot pulse.png --debug-video rois.mp4
```

### Phase 1 on a dataset

```bash
# 1. Face landmarks + ROI averaging (slow, ~10-30 ms/frame; resumable; run once per dataset)
python scripts/extract_traces.py ubfc data/UBFC-rPPG --out cache/traces/ubfc --workers 4

# 2. All classical methods x ROIs, scored with the frozen protocol (seconds)
python scripts/run_classical.py cache/traces/ubfc --out results/classical/ubfc --window 10

# Test subjects only (to compare with the deep models later):
python scripts/run_classical.py cache/traces/ubfc --split splits/ubfc.json --subset test --out results/classical/ubfc_test
```

`run_classical.py` writes a per-video CSV (predicted/GT HR, SNR, runtime, dataset labels) and
a summary CSV (MAE, RMSE, MAPE, Pearson, SNR per method and ROI). The `face_box` ROI is the
toolbox-style enlarged face crop, so the Phase 2 ROI ablation is already in these outputs.

Add `--debug-videos cache/debug/ubfc` to step 1 to save ROI-overlay videos for the report.

### Splits

```bash
python scripts/make_splits.py                                  # already done; files in splits/
python scripts/make_splits.py --check ubfc data/UBFC-rPPG      # verify against the data on disk
```

Subject-independent, ~70/15/15, seed 2025: UBFC 30/6/6, PURE 6/2/2, MMPD 23/5/5 subjects.
Do not regenerate after experiments start.

### Smoke test without real data

```bash
python scripts/make_synthetic_dataset.py face.jpg --out cache/synthetic-ubfc
python scripts/extract_traces.py ubfc cache/synthetic-ubfc --out cache/traces/synthetic
python scripts/run_classical.py cache/traces/synthetic --out cache/results/synthetic
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

The toolbox's own post-processing hard-codes a 0.6-3.3 Hz band, so in Phase 2/3 toolbox
outputs will be re-scored with `rppg.protocol` rather than used as-is.

## Tests

```bash
pytest                                              # 65 fast tests, ~3 s
RPPG_FACE_IMAGE=face.jpg pytest -m slow             # end-to-end through MediaPipe on a synthetic pulsing face
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
  pipeline.py          trace extraction (cached) + estimation
  synth.py             synthetic traces / videos with known HR
  cli.py               `rppg` command
scripts/               make_splits, extract_traces, run_classical, make_synthetic_dataset
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
