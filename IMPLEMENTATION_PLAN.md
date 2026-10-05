# Implementation Plan — Contactless Vital Sign Estimation from Facial Video (rPPG)

BCSE417L Machine Vision · Mayank (23BAI0158), Vyomesh Pandita (23BAI0151)

This plan turns the Review 2 document into concrete, phased engineering work. It is
organised so that every promise in the report maps to a deliverable:

| Report promise | Delivered in |
|---|---|
| Working prototype: RGB video → BPM | Phase 1 (offline), Phase 6 (live webcam) |
| Classical methods CHROM / POS / ICA | Phase 1 (own implementation), Phase 2 (toolbox cross-check) |
| Deep models PhysNet / TS-CAN / EfficientPhys / PhysFormer | Phase 3 |
| Benchmark on UBFC-rPPG, PURE, MMPD with MAE/RMSE/MAPE/Pearson/SNR | Phases 2–3 |
| Cross-dataset generalisation | Phase 3 |
| Condition-wise robustness on MMPD (skin tone, motion, lighting) | Phase 4 |
| Accuracy-vs-efficiency trade-off (latency, FPS, params, memory) | Phase 5 |
| Reproducible protocol | Phase 0 (pinned env, fixed splits), Phase 7 |

---

## Guiding decisions

1. **Two codebases, clearly separated.**
   - **Our own pipeline** (`src/rppg/`): MediaPipe face landmarks → forehead/cheek
     ROI masks → RGB traces → CHROM/POS/ICA → HR. This is the "machine vision"
     contribution, the thing we can explain line by line, and the backbone of the
     live demo.
   - **rPPG-Toolbox** (pinned git submodule under `external/`): used for the deep
     models and as the standardised benchmark harness. We do not modify its
     internals; we only add our own YAML configs and post-processing scripts.
2. **Our classical results must be checked against the toolbox's classical results.**
   If our POS on UBFC-rPPG is far from the toolbox's POS, there's a bug in ours.
3. **One evaluation definition everywhere.** Same HR band, same window length, same
   HR estimator (FFT), same ground-truth HR derivation, for every method. Results are
   saved **per video** (not only as averages) so any table, plot or condition-wise
   breakdown can be regenerated without re-running models.
4. **Get access to datasets on day 1.** PURE and MMPD need request forms/licence
   agreements and can take days to weeks. UBFC-rPPG is openly downloadable, so all
   early work happens on UBFC.

---

## Proposed repository layout

```
vision/
├── IMPLEMENTATION_PLAN.md
├── README.md                  # setup + how to reproduce every table
├── environment.yml            # pinned versions (python, torch, mediapipe, opencv, scipy…)
├── external/rPPG-Toolbox/     # git submodule, pinned to a specific commit
├── configs/toolbox/           # our YAML configs for the toolbox (one per experiment)
├── splits/                    # fixed subject-ID splits as JSON (committed)
├── data/                      # gitignored; symlinks to raw datasets
├── src/rppg/
│   ├── io/                    # video/frame readers, UBFC / PURE / MMPD loaders + GT
│   ├── face/                  # MediaPipe landmarker wrapper, ROI polygon masks
│   ├── signal/                # preprocess.py, chrom.py, pos.py, ica.py, green.py, hr.py
│   ├── eval/                  # metrics.py, snr.py, bland_altman.py, stats.py
│   └── bench/                 # latency.py, complexity.py (params, FLOPs, memory)
├── scripts/                   # run_classical.py, run_toolbox.sh, collect_results.py,
│                              # mmpd_conditions.py, benchmark_efficiency.py
├── app/live_demo.py           # webcam prototype
├── tests/                     # unit tests (synthetic signals, metrics)
├── results/                   # per-video CSVs, summary tables, figures (small files only)
└── notebooks/                 # exploration + figure generation
```

---

## Phase 0 — Setup, data access, protocol freeze (Week 1)

**Goal:** everything needed to start running experiments, with zero ambiguity about
how they'll be evaluated.

Tasks
- [ ] **Request PURE and MMPD access immediately** (PURE: request form on the TU
      Ilmenau page; MMPD: sign the licence agreement in the GitHub repo. Ask for
      **mini-MMPD (320×240)**, which is what the toolbox configs target and is far
      smaller than full resolution). Download UBFC-rPPG (DATASET_2).
- [ ] Create the repo skeleton above; add rPPG-Toolbox as a submodule and **record the
      commit hash** in the README.
- [ ] Build the environment following the toolbox's `setup.sh`, then add our extras
      (mediapipe, pytest, pandas, seaborn, thop/fvcore). Pin everything in
      `environment.yml`.
- [ ] Secure GPU compute (lab GPU, Colab, or Kaggle). Measure disk space: the toolbox
      caches preprocessed clips, which can be tens of GB per dataset/config.
- [ ] **Freeze the evaluation protocol** in `README.md` (see "Evaluation protocol" below).
- [ ] **Freeze the subject splits**: write `splits/ubfc.json`, `splits/pure.json`,
      `splits/mmpd.json` (70/15/15 by subject, fixed seed, committed to git).

Exit criteria
- `python -c "import torch, mediapipe, cv2, scipy"` works; toolbox runs its unit
  example on UBFC with a pretrained checkpoint.
- Split files and protocol committed.

---

## Phase 1 — Own classical pipeline on UBFC-rPPG (Weeks 2–3)

**Goal:** a working end-to-end offline prototype (video file → BPM) built by us,
implementing Blocks 1–4, 5A, 6, 7, 8, 9 of the report.

Tasks
1. **Block 1 — Input.** `io/`: frame reader (OpenCV) returning frames + timestamps;
   UBFC loader that also parses `ground_truth.txt` (BVP, HR, timestamps).
2. **Block 2 — Face detection.** `face/landmarker.py`: wrap MediaPipe face landmarks.
   Note: the legacy `mp.solutions.face_mesh` API (468 landmarks) is being superseded
   by the Tasks `FaceLandmarker` API (478 landmarks incl. iris); pick one, pin the
   mediapipe version, and keep the wrapper interface stable. Handle missed detections
   (reuse last landmarks for ≤ N frames, otherwise flag the frame).
3. **Block 3 — ROI extraction.** `face/roi.py`: landmark-index polygons for forehead,
   left cheek, right cheek; `cv2.fillPoly` masks; exclude eyes/brows/lips/nostrils.
   Save a debug video with ROI overlays — this is a great figure for the report.
4. **Block 4 — Preprocessing.** `signal/preprocess.py`: per-frame spatial mean of
   R, G, B over the mask → `(T, 3)` trace; resample to a uniform rate if timestamps
   are jittery; temporal normalisation; detrending (smoothness-priors detrend or
   `scipy.signal.detrend`); Butterworth band-pass with `filtfilt`.
5. **Block 5A — Classical methods.** `signal/chrom.py`, `pos.py`, `ica.py`, plus
   `green.py` as the simplest baseline. POS and CHROM use overlapping windows
   (~1.6 s) with overlap-add, as in the original papers. ICA via
   `sklearn.decomposition.FastICA`, selecting the component with the highest spectral
   peak in the HR band.
6. **Blocks 6–7 — Pulse + HR.** `signal/hr.py`: FFT/Welch peak in band → BPM; and
   peak detection (`find_peaks`) → mean IBI → BPM.
7. **Block 9 — Metrics.** `eval/metrics.py`: MAE, RMSE, MAPE, Pearson r; `eval/snr.py`:
   SNR as in de Haan & Jeanne (power around the GT HR fundamental + first harmonic
   vs. the rest of the band).
8. **Tests.** `tests/`: synthetic RGB traces with a known 72 BPM pulse + noise + drift
   → every method must recover HR within ±2 BPM; metric functions checked against
   hand-computed values.
9. `scripts/run_classical.py --dataset ubfc --method pos` → writes
   `results/classical/ubfc_pos.csv` (one row per video: id, HR_pred, HR_gt, SNR,
   runtime).

Deliverables
- Offline prototype: `python -m rppg.cli video.avi` prints BPM and plots the pulse.
- First results table: GREEN / ICA / CHROM / POS on UBFC-rPPG (our implementation).

Exit criteria
- All tests pass. POS and CHROM MAE on UBFC-rPPG in the low single digits of BPM
  (the report's ~4 BPM ballpark). If it's much worse, debug before moving on
  (common culprits: wrong band, missing normalisation, BVP/video misalignment).

---

## Phase 2 — Toolbox classical baselines on all three datasets (Week 3–4)

**Goal:** standardised unsupervised results on UBFC-rPPG, PURE and MMPD, and a
cross-check of our implementation.

Tasks
- [ ] Write toolbox configs for the unsupervised methods (ICA, POS, CHROM, GREEN;
      LGI/PBV are free extras) on each dataset.
- [ ] Turn on saving of test outputs and write `scripts/collect_results.py` that
      converts the toolbox outputs into the **same per-video CSV format** as Phase 1.
- [ ] Run our own pipeline on PURE and MMPD too (loaders for PURE's PNG frames +
      JSON ground truth, and MMPD's `.mat` files).
- [ ] **Cross-check table:** ours vs. toolbox for POS/CHROM/ICA on each dataset.
- [ ] **ROI ablation (cheap, adds novelty):** toolbox-style face bounding-box crop vs.
      our MediaPipe forehead+cheek masks, same method (POS), same datasets.

Deliverables
- Table: classical methods × {UBFC-rPPG, PURE, MMPD} × {MAE, RMSE, MAPE, r, SNR}.
- Table: ROI strategy ablation.

Exit criteria
- Our implementation and the toolbox agree within a small margin, or the difference
  is understood and explained (e.g. different ROI, different detrending).

---

## Phase 3 — Deep-learning models (Weeks 4–7)

**Goal:** PhysNet, TS-CAN, EfficientPhys, PhysFormer evaluated intra- and cross-dataset
under one protocol. (DeepPhys is optional — near-free in the toolbox and is the
ancestor of TS-CAN/EfficientPhys.)

Step 3a — Pretrained inference first (Week 4)
- The toolbox ships pretrained checkpoints (trained on PURE, UBFC-rPPG, SCAMPS, …).
  Run inference-only configs with them on all three datasets. This produces a full
  first DL table quickly and validates the data pipeline before spending GPU hours.

Step 3b — Training under our protocol (Weeks 5–7)
- **Intra-dataset:** train/val/test with the fixed 70/15/15 subject splits from Phase 0.
  UBFC-rPPG has only ~42 subjects, so test-set results will be noisy; report
  confidence intervals (Phase 7).
- **Cross-dataset:** train on PURE → test on UBFC-rPPG and MMPD; train on UBFC-rPPG →
  test on PURE and MMPD. This is the toolbox's standard protocol, so our numbers can
  be compared directly with its published tables.
- Use the toolbox's default preprocessing per model (input size, `DiffNormalized` /
  `Standardized`, chunk length) — don't hand-tune per model, or the comparison is
  no longer fair.
- Model selection on the validation set only; log every run (config, seed, epoch,
  val loss) to `results/runs.csv`.
- Fix seeds; if compute allows, repeat the main cross-dataset runs with 3 seeds.

Compute budget tips
- Train one model end-to-end first (TS-CAN or EfficientPhys: cheapest) to shake out
  problems; PhysFormer is the most expensive, so schedule it last.
- Preprocessed caches can be reused across models that share an input format.

Deliverables
- Master table: {GREEN, ICA, CHROM, POS, PhysNet, TS-CAN, EfficientPhys, PhysFormer}
  × {intra UBFC, intra PURE, PURE→UBFC, UBFC→PURE, →MMPD}.
- Saved checkpoints and per-video prediction CSVs.

Exit criteria
- All cells of the master table filled; numbers are in the same ballpark as the
  toolbox paper for matching protocols (or differences are explained).

---

## Phase 4 — Condition-wise robustness analysis on MMPD (Weeks 7–8)

**Goal:** the report's main novelty claim — where and why methods fail.

Tasks
- [ ] Extract MMPD metadata per video (skin tone, motion, lighting, exercise, glasses,
      hair cover, make-up — stored in each `.mat` file) into
      `results/mmpd_metadata.csv`.
- [ ] `scripts/mmpd_conditions.py`: join per-video predictions (all methods) with
      metadata; compute metrics grouped by
      - skin tone (Fitzpatrick type),
      - motion (stationary / rotation / talking / walking),
      - lighting (LED-low / LED-high / incandescent / natural),
      - and selected 2-way combinations (e.g. skin tone × lighting).
- [ ] Figures: grouped bar charts of MAE per condition per method; heatmap of
      method × condition MAE; Bland–Altman plots for the best classical and best DL
      method.
- [ ] Report group sizes next to every number; with ~20 videos per subject, small
      groups need confidence intervals before drawing conclusions.
- [ ] Write the analysis: which method family degrades least, under which condition,
      and the likely physical reason (melanin absorption → lower pulsatile amplitude;
      motion → specular/intensity changes; low light → sensor noise).

Deliverables
- Robustness tables + figures, and a written discussion section for the report.

---

## Phase 5 — Efficiency benchmarking (Week 8–9, can run in parallel with Phase 4)

**Goal:** accuracy-vs-compute trade-off across method families.

Tasks
- [ ] `bench/complexity.py`: parameter count (`sum(p.numel())`), FLOPs per clip
      (thop or fvcore), normalised to per-frame.
- [ ] `bench/latency.py`: warm-up then timed runs on **CPU and GPU** separately
      (`torch.cuda.synchronize()` around GPU timings); report ms/frame, FPS, and peak
      memory (`torch.cuda.max_memory_allocated`; `tracemalloc`/`psutil` for CPU).
      Clip-based models (PhysNet, PhysFormer) process a whole chunk at once — divide
      by chunk length so per-frame numbers are comparable to frame-based models.
- [ ] Time the classical pipeline the same way, and report **face detection/ROI
      cost separately** from the rPPG method cost (otherwise MediaPipe dominates and
      hides the difference between CHROM and POS).
- [ ] Record the exact hardware (CPU model, GPU model, RAM) in the table caption.
      The report quotes Raspberry Pi 4B and literature numbers; we report what we
      actually measured and cite literature numbers separately.
- [ ] Plot: MAE (on MMPD and on UBFC) vs. ms/frame, log-x; mark the Pareto front.

Deliverables
- Efficiency table + accuracy-vs-latency scatter plot.

---

## Phase 6 — Live webcam prototype (Weeks 9–10)

**Goal:** demonstrate the system working on a real person.

Design
- Capture with OpenCV; attempt to lock auto-exposure / auto-white-balance (a common
  source of fake "pulses"); log real timestamps and resample to a uniform rate.
- Sliding window: ~10 s buffer, BPM update every 1 s; show HR only once the buffer
  is full.
- Default engine: our MediaPipe + POS pipeline (runs in real time on CPU).
  Optional toggle: a lightweight DL model (TS-CAN or EfficientPhys) using a
  pretrained checkpoint.
- UI (Streamlit or plain OpenCV window): video with ROI overlay, live pulse
  waveform, spectrum, BPM, and a signal-quality indicator (SNR or spectral peak
  ratio) that warns when the estimate is unreliable (motion, low light, no face).

Validation
- Record a few short sessions with a fingertip pulse oximeter or smartwatch reading
  as rough reference (rest vs. after light exercise). Present this only as a
  qualitative demo, not as a benchmark result.

Deliverables
- `app/live_demo.py`, a screen recording for the final review.

---

## Phase 7 — Statistical analysis, reproducibility, final report (Weeks 10–12)

Tasks
- [ ] Bootstrap 95% CIs for MAE and r for every main table cell.
- [ ] Paired significance tests between methods on the same videos (Wilcoxon
      signed-rank on absolute errors) for the headline comparisons.
- [ ] `README.md`: one command per table/figure; all configs in `configs/`; split
      files in `splits/`; toolbox commit hash; hardware used.
- [ ] Final report: results, robustness discussion, efficiency trade-off,
      limitations, future work (respiration/HRV, motion-robust ROI tracking,
      domain adaptation).
- [ ] Final demo + slides.

---

## Evaluation protocol (freeze in Phase 0)

| Item | Decision to record |
|---|---|
| HR frequency band | One band for every method, e.g. 0.7–3.0 Hz (42–180 BPM) as in the report. The toolbox's default post-processing band is narrower, so set it explicitly in every config. |
| Evaluation window | Whole-video HR (simplest, matches most papers), and optionally 10 s / 30 s windows for a secondary table. Same choice for every method. |
| HR estimator | FFT peak as primary; peak-detection HR as a secondary column. |
| Ground-truth HR | Derived from the reference BVP with the same estimator and window as the prediction (not the device's displayed HR). |
| Metrics | MAE, RMSE, MAPE, Pearson r computed over per-video HR; SNR averaged over videos. |
| Splits | Subject-independent 70/15/15, fixed JSON files; cross-dataset = train on one dataset, test on all videos of another. |
| Seeds | Fixed and logged; ≥ 3 seeds for headline DL runs if compute allows. |

---

## Suggested division of work

| Track | Owner (suggested) | Phases |
|---|---|---|
| A — Vision + classical + demo | Mayank | 1, 2 (own pipeline, ROI ablation), 6 |
| B — Deep learning + benchmarking | Vyomesh | 2 (toolbox runs), 3, 5 |
| Shared | Both | 0, 4, 7 |

Swap freely; the point is that Tracks A and B can progress in parallel after Phase 0.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| PURE/MMPD access is delayed | Request on day 1; do Phases 1–3a on UBFC-rPPG in the meantime. |
| Not enough GPU time | Use pretrained checkpoints (Phase 3a) for a complete table; train the cheapest models first; drop extra seeds before dropping models. |
| Disk space for toolbox caches | Use mini-MMPD; delete caches for finished configs; cache only the input formats you need. |
| MediaPipe API changes | Pin the version; isolate it behind `face/landmarker.py`. |
| Our classical results disagree with the toolbox | Phase 2 cross-check is designed to catch this early; check band, normalisation, BVP alignment, PURE's 60 Hz ground truth vs. 30 fps video. |
| Small test sets → noisy conclusions | Report CIs and group sizes; lean on cross-dataset results, which use the full target dataset. |
| Webcam demo unstable | Good, steady lighting; lock exposure; signal-quality gate; keep a pre-recorded fallback video. |

---

## Things to correct or clarify in the report before the next review

1. **Face detection in the DL branch.** The architecture says MediaPipe Face Mesh
   feeds both branches, but rPPG-Toolbox's dataloaders use their own face detector
   (Haar cascade or RetinaFace) and bounding-box crops. Either state that the DL branch
   uses toolbox preprocessing, or feed it MediaPipe-based crops. The ROI ablation in
   Phase 2 turns this difference into a result.
2. **"Same preprocessed data" for both branches.** Classical methods consume RGB
   traces; DL models consume cropped frame clips. The accurate claim is "same
   videos, same splits, same face crops where applicable, same HR post-processing".
3. **MMPD skin-tone range.** The report says Fitzpatrick I–VI; check the MMPD paper
   and metadata (I believe it covers types III–VI) and correct if needed.
4. **POS citation year.** The literature survey says Wang et al., 2017, the reference
   list says 2016. The IEEE TBME 64(7) issue is 2017; make them consistent.
5. **Literature numbers need a protocol label.** E.g. "PhysNet ~0.58 BPM MAE on
   UBFC" — state whether intra- or cross-dataset and from which source, since the
   same model gives very different numbers under different protocols.
6. **Efficiency claims.** Replace hardware-specific literature figures (Raspberry
   Pi 4B, 6 ms/frame) in the "expected" section with "to be measured on <our
   hardware>", and cite literature figures separately.
