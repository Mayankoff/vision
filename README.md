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
| 0. Setup and data access | done: package, settings, loaders (UBFC-rPPG, PURE, synthetic), [data guide](docs/DATA.md), Windows scripts. Dataset requests are the team's to send |
| 1. Multi-patch core (steps 1–7) | built and tested on synthetic data; the exit check (beat single-region POS on UBFC-rPPG and PURE) needs the datasets |
| 2–7 | not started |

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
afscreen evaluate cache/traces/ubfc                   # every variant vs the reference -> results/phase1/ubfc
```

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
