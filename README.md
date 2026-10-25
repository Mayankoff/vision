# Quality-Aware Multi-Patch rPPG for Atrial Fibrillation Screening

BCSE417L Machine Vision course project. Mayank (23BAI0158), Vyomesh Pandita (23BAI0151).

From an ordinary RGB face video, the system measures the pulse in up to 25 small skin patches and
trusts each patch only when its signal is reliable. It then screens for **atrial fibrillation**
(primary output) and flags **slow, fast or extra heartbeats** (secondary output), each with a
confidence. See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the plan, and the
[simplified](docs/figures/block_diagram_simple.png) and
[detailed](docs/figures/block_diagram_proposed.png) architecture diagrams.

## Status

| Phase | State |
|---|---|
| 0. Setup and data access | done. Data used: **UBFC-rPPG** (partial download: 12 face videos, 7 with a reference) and **MIMIC PERform AF** (complete). PURE dropped (no time for the access request); see [data guide](docs/DATA.md) |
| 1. Multi-patch core (steps 1–7) | built; evaluated on 7 UBFC videos with a reference ([results](results/phase1/ubfc_all/summary.md)): HR MAE 0.2 BPM for every variant, beat F1 0.98–0.99. These still, well-lit videos are too easy for the trust checks to beat the single-region baseline on HR (a tie) |
| 3. AF screening (step 10) | built; [MIMIC results](results/phase3/mimic/summary.md): random forest, leave-one-subject-out, AUC 0.96 at a camera-like 30 fps, 19/19 AF and 14/16 non-AF subjects correct. On the 12 normal-rhythm face videos the full method gives 1/12 false alarms vs 2/12 for the single region ([results](results/phase3/face/summary.csv)) |
| 4. Extra beats, slow / fast flags (step 11) | built; [summary](results/phase4/SUMMARY.md). MIT-BIH (tuned on DS1, tested on DS2, 16 ms camera-like timing noise): 81% of ventricular and 15% of atrial extra beats found, 86% precision. Removing extra beats nearly halves AF false alarms on MIT-BIH and brings the face videos to 0/12. Slow/fast flags agree with the ECG in 95% of 2-min MIMIC segments (kappa 0.88) |
| 2, 5–7 | not started |

The code of the earlier version of the project (classical methods, rPPG-Toolbox integration) was
removed; it is still in the git history.

## What is built (Phase 1)

```
video -> MediaPipe: 478 3D landmarks + head pose                  face/tracker.py
      -> 25 skin patches (merged to 12 or 6 when the face is small) face/patches.py, patches_def.py
      -> colour traces per patch, cached                          traces.py
      -> motion removal on colours (only when the head moves)     core/fusion.py
      -> POS per patch                                            signal/pos.py
      -> trust per patch, per 8 s window:                         core/trust.py
         orientation x brightness x consistency^2 x beat agreement
      -> trust-weighted fusion (+ second motion pass) + confidence core/fusion.py
      -> heart rate (spectrum and beats), beat times              pipeline.py, signal/beats.py
```

- **Rhythm-independent quality:** trust and confidence never assume a regular rhythm, so a
  clean AF pulse keeps high confidence. Confidence is the geometric mean of four parts:
  - patch consistency,
  - beat-shape consistency,
  - low head motion,
  - enough trusted patches.
- **Ablation variants**, one ingredient added at a time: `single` (one skin region, plain POS),
  `patch_mean`, `+facing`, `+brightness`, `+consistency`, `+beats`, `full`.
- **Synthetic data** (`synth.py`, `scripts/make_synthetic_videos.py`) with known beat times:
  regular, AF-like, slow, fast, extra beats, small faces, head sway.

Results on synthetic videos (9 videos; a check of the machinery, not a benchmark) are in
`results/phase1/synthetic/summary.md`.

## Setup

**Windows (PowerShell, Python 3.11–3.13):**

```powershell
git pull
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
.venv\Scripts\Activate.ps1
```

`setup.ps1` reuses an existing `.venv` and removes the old `rppg` package. It then:
1. installs the requirements and the `afscreen` package,
2. downloads the MediaPipe face model (3.7 MB, to `models\`),
3. runs the tests.

The old rPPG-Toolbox environment (`.venv-toolbox`) is no longer used and can be deleted.

<details><summary>Linux / macOS</summary>

```bash
sudo apt-get install -y libegl1 libgles2      # Linux only: MediaPipe needs EGL / GLES
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q
```
</details>

## Usage

```bash
afscreen run my_face.mp4 --debug-video trust.mp4      # one video: HR, beats, confidence; trust map video
afscreen extract ubfc data/UBFC-rPPG --workers 4      # cache patch traces for a dataset
python scripts/extract_from_zips.py datasets/UBFC     # ... or straight from the Google Drive zip parts
afscreen evaluate cache/traces/ubfc_all               # every variant vs the reference -> results/phase1/...
afscreen af-train datasets/MIMIC                      # AF classifier on MIMIC PERform AF -> results/phase3/mimic
afscreen af-screen cache/traces/ubfc_all cache/traces/noref   # AF false alarms on face videos -> results/phase3/face
afscreen extra-beats "datasets/MIT BIH" --mimic datasets/MIMIC   # extra-beat rule + its effect on AF -> results/phase4
afscreen hr-flags datasets/MIMIC                      # slow / fast flags vs ECG -> results/phase4/hr_flags
```

`af-screen` removes extra beats first when `results/phase4/mitbih` exists (`--extra-beats ''` turns it off).

- **Phase 1 check on Windows:** `scripts\windows\run_phase1.ps1 -Dataset ubfc` (or `-Dataset pure`).
- **Datasets:** see [docs/DATA.md](docs/DATA.md).
- **Test without a dataset:**
  1. make synthetic videos from any frontal face photo: `python scripts/make_synthetic_videos.py face.jpg`,
  2. run `afscreen extract synthetic data/synthetic`.
- **Phone videos:** record in H.264 ("Most Compatible" on iPhone). OpenCV usually cannot read HEVC.

## Tests

```bash
pytest -q                                              # unit tests (~45 s)
AFSCREEN_FACE_IMAGE=face.jpg pytest -q -m slow         # end-to-end through MediaPipe on synthetic videos
```

## Repository layout

```
src/afscreen/        the package (config.py holds every setting)
scripts/             make_patches.py, make_synthetic_videos.py, download_data.py, figure_block_diagram.py, windows/
tests/               unit and end-to-end tests
docs/                DATA.md, figures (architecture diagrams, patch levels)
data/ cache/         datasets and cached traces (gitignored)
results/             evaluation outputs
```
