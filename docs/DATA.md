# Data

Datasets are never committed: `data/` and `datasets/` are gitignored.

## What the project uses

| Dataset | Used for (phase) | What we have | Folder |
|---|---|---|---|
| **MIMIC PERform AF** | train / validate the AF classifier (Phase 3) | complete: 19 AF + 16 non-AF subjects, 20 min PPG + ECG at 125 Hz ([Zenodo](https://zenodo.org/record/6950488)) | `datasets/MIMIC/` (the CSV zips are read directly) |
| **UBFC-rPPG DATASET_2** | heart rate, beat timing; AF false alarms on normal rhythm | **partial**: 7 of the Google Drive zip parts. 7 videos, of which 2 (subjects 25, 26) have their reference; 5 (1, 4, 5, 27, 48) do not | `datasets/UBFC/dataset 2/` |
| **UBFC-rPPG DATASET_1** | same | 5 videos with a reference: 8, 10, 11, and two loose files identified as 12 (`vid-002.avi`) and 5 (`vid-003.avi`) by heart rate and duration (exact match to the reference) | `datasets/UBFC/dataset 1/` |
| **MIT-BIH Arrhythmia** | extra-beat detector (Phase 4) | complete (Kaggle CSV version: 48 records, beat annotations used; the 4 paced records are excluded). The PhysioNet WFDB version (`python scripts/download_data.py mitdb`) also loads | `datasets/MIT BIH/` (zip read directly) |
| Synthetic videos | end-to-end tests with known beat times | `python scripts/make_synthetic_videos.py face.jpg` (any frontal face photo, e.g. a UBFC frame) | `data/synthetic/` |

So there are **12 normal-rhythm face videos**. 7 have a reference pulse (Phase 1 heart-rate and beat
evaluation), and all 12 count for the AF false-alarm rate, since every UBFC subject is healthy.

### Not used, and what that costs

| Dataset | Why not | Consequence for the report |
|---|---|---|
| PURE | access request; no time | no controlled head-motion test of motion removal (only synthetic sway videos) |
| OBF | access request | no sensitivity on real AF *face videos*; AF detection is shown on MIMIC PPG instead (plan, Section 4) |
| rest of UBFC-rPPG DATASET_2 | download failed | small face-video test set: report per-video results and treat them as a pilot |
| UBFC-Phys, COHFACE | optional (talking videos; breathing reference) | no talking false-alarm test; breathing rate (Phase 2) cannot be validated against a belt |

The UBFC videos are about 1 minute long, so each gives 1–2 AF windows of 30 s (20 in total), and
the plan's 2-minute rule for the slow / fast heart-rate flag cannot be applied to them.

## Caching traces from the zip parts

Drive splits a folder into parts that each hold about one video plus *other* subjects' reference
files. `scripts/extract_from_zips.py` reads every reference from all parts, unpacks one video at a
time to the temp folder, caches its traces and deletes the video again (each is 1–2 GB):

```bash
python scripts/extract_from_zips.py datasets/UBFC --video "datasets/UBFC/dataset 1/vid-002.avi=12-gt" --video "datasets/UBFC/dataset 1/vid-003.avi=5-gt"
```

It writes `cache/traces/ubfc/` (DATASET_2 with a reference), `cache/traces/ubfc1/` (DATASET_1) and
`cache/traces/noref/` (no reference). `cache/traces/ubfc_all/` is the first two together.

## Notes on the UBFC references

- DATASET_2 frame rates are really ~23 fps for some subjects (e.g. 25, 26): the reference
  timestamps agree with the video header.
- DATASET_1 `gtdump.xmp` is CSV: time (ms), HR, SpO2, PPG at ~60 Hz; there are no frame
  timestamps, so the PPG is stretched over the video (durations agree to within 0.4 s).
- The camera pulse is **inverted** relative to the oximeter PPG and arrives 0.10–0.21 s later
  (pulse transit time). The evaluation aligns sign and delay per video before matching beats
  (`eval/reference.py: align_reference`); heart rate and IBIs are unaffected either way.

## Expected layouts (for the loaders)

```
data/UBFC-rPPG/subject1/vid.avi
data/UBFC-rPPG/subject1/ground_truth.txt        # line 1: PPG, line 2: HR, line 3: time (s)
data/UBFC-1/8-gt/vid.avi + gtdump.xmp           # DATASET_1 (afscreen extract ubfc1 data/UBFC-1)
data/PURE/01-01/01-01.json                      # loader kept, dataset not used
data/PURE/01-01/01-01/Image<timestamp>.png
data/synthetic/<case>/vid.avi + truth.npz       # exact beat times
data/MIT-BIH/100.dat, 100.hea, 100.atr, ...
```
