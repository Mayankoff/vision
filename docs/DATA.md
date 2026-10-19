# Data

Raw data goes under `data/` (gitignored; never commit datasets). Request the gated datasets
**early**: approval can take weeks, and OBF decides how strong the final AF result can be.

| Dataset | Used for (phase) | How to get it | Folder |
|---|---|---|---|
| **UBFC-rPPG** (DATASET_2) | heart rate and beat timing; Phase 1 exit check | download from the authors' page (Bobbia et al., 2019) | `data/UBFC-rPPG/` |
| **PURE** | same, with controlled head motion (motion removal) | request from TU Ilmenau (Neuroinformatics and Cognitive Robotics lab) | `data/PURE/` |
| **MIMIC PERform AF** | train/validate the AF classifier (Phase 3) | public; see the `ppg-beats` documentation (Charlton et al., 2022) | `data/MIMIC-PERform-AF/` |
| **MIT-BIH Arrhythmia** | extra-beat detector (Phase 4) | `python scripts/download_data.py mitdb` | `data/MIT-BIH/` |
| **OBF** | AF on real face videos (Phase 3) | request from the University of Oulu (Li et al., 2018) | `data/OBF/` |
| **UBFC-Phys** | talking / stress-task videos: AF false alarms | IEEE DataPort ("UBFC-Phys") | `data/UBFC-Phys/` |
| **COHFACE** | breathing rate (Phase 2) | request from Idiap | `data/COHFACE/` |
| Synthetic videos | end-to-end tests with known beat times | `python scripts/make_synthetic_videos.py face.jpg` (any frontal face photo) | `data/synthetic/` |

Loaders exist for UBFC-rPPG, PURE and the synthetic videos (`src/afscreen/io/datasets.py`); the
others are added in the phase that first uses them.

## Expected layouts

```
data/UBFC-rPPG/subject1/vid.avi
data/UBFC-rPPG/subject1/ground_truth.txt        # line 1: PPG, line 2: HR, line 3: time (s)
data/PURE/01-01/01-01.json
data/PURE/01-01/01-01/Image<timestamp>.png
data/synthetic/<case>/vid.avi + truth.npz       # exact beat times
data/MIT-BIH/100.dat, 100.hea, 100.atr, ...
```

## Phase 1 check (any dataset with a loader)

```bash
afscreen extract ubfc data/UBFC-rPPG --workers 4      # MediaPipe + patch traces, cached in cache/traces/ubfc
afscreen evaluate cache/traces/ubfc                   # every variant vs the reference -> results/phase1/ubfc
```

On Windows: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run_phase1.ps1 -Dataset ubfc`.
