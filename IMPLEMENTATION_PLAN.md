# Implementation Plan — Quality-Aware Multi-Patch rPPG for Atrial Fibrillation Screening

BCSE417L Machine Vision · Mayank (23BAI0158), Vyomesh Pandita (23BAI0151)

> **Status:** Phases 0 and 1 are built (package `afscreen`, see README) and tested on synthetic
> data; the Phase 1 exit check on UBFC-rPPG and PURE needs the datasets. The code from the
> earlier project was removed. Existing methods (POS, CHROM, PhysNet, …) are not part of the architecture; they are only
> **comparison baselines** in the evaluation. The system needs **one ordinary RGB camera**
> (webcam or phone); no extra hardware.

Architecture diagrams:
- simplified (for slides): [`docs/figures/block_diagram_simple.png`](docs/figures/block_diagram_simple.png) (black and white: `block_diagram_simple_bw.png`)
- detailed (for the report): [`docs/figures/block_diagram_proposed.png`](docs/figures/block_diagram_proposed.png) (black and white: `block_diagram_proposed_bw.png`)

---

## 1. What we are building

A contactless system that takes **RGB video of a face** and outputs:

| Output | Role |
|---|---|
| **AF screening result**: "regular rhythm", "possible AF — see a doctor", or "signal unreliable" | **primary** real-life output (Section 3) |
| **Heart-rate flags**: "slow heart rate", "fast heart rate" (at rest), "extra beats" | **secondary** real-life output, from the same heart rate and beat timings (Section 3) |
| Heart rate (BPM), breathing rate, HRV (RMSSD, SDNN) | vital signs, and the inputs to the two analyses |
| Confidence | the system says when it cannot be trusted instead of giving a wrong answer |

### Pipeline (matches the detailed diagram)

1. **Face landmarks + head pose:** MediaPipe FaceLandmarker gives 478 3D landmarks and a head rotation/translation per frame.
2. **Skin patches:** forehead and cheeks are split into patches made of face-mesh triangles, so they move with the face.
   - **The patch count adapts to face size:**
     - **25 patches** (9 forehead, 8 per cheek) when the face is large, about 200 px wide or more, as in 640×480 dataset videos,
     - **merged to 12 or 6** when the face is smaller, so each patch keeps at least about 300–400 skin pixels.
   - Averaging over pixels is what cancels camera noise, and the pulse is well under 1% of skin brightness. The pixel minimum is our own rule of thumb; the patch-count ablation (Phase 5) checks it.
3. **Patch colour traces:** mean R, G, B of every patch in every frame.
4. **POS per patch:** one candidate pulse signal per patch.
5. **Trust score per patch**, over 8 s windows updated every second. The four checks are designed **not to assume a regular rhythm**, so real AF is not mistaken for noise:
   - **face orientation:** cosine of the angle between the patch's 3D surface normal and the camera direction (Lambert's law); weight = cos θ, and 0 when cos θ < 0.3 (turned more than about 72°),
   - **brightness:** luma Y = 0.299R + 0.587G + 0.114B; weight 0 below 20 (deep shadow) or above 245 (shine or saturation), with smooth ramps between 20–45 and 220–245,
   - **patch consistency:** correlation of the patch's pulse with the fused pulse of all *other* patches (leave-one-out). In AF all patches still share the same irregular beats; noise differs from patch to patch,
   - **beat agreement:** share of the patch's detected beats that fall within about 100 ms of the consensus beats.

   trust = orientation × brightness × consistency × beat agreement, normalised to sum to 1.
6. **Weighted fusion + confidence:**
   - **Fused pulse:** Σ trust_k × pulse_k.
   - **Confidence (0–1)** for each window combines four rhythm-independent measures:
     1. mean consistency of the trusted patches,
     2. **beat-shape consistency:** correlation of each beat with the window's average beat, a standard PPG quality index [19],
     3. head-motion level,
     4. effective number of trusted patches.
   - The gating threshold is chosen on validation data, at the point where beat-timing error starts to rise.
   - *Why not spectral sharpness:* an irregular rhythm has no single frequency, so a "sharp spectral peak" confidence would reject exactly the AF windows we need (tested in Phase 5).
7. **Motion removal:** the 6 head-pose signals (yaw, pitch, roll, x, y, z) and their rates of change are band-passed like the pulse. A **ridge regression** on them is subtracted in two places:
   - **from every patch's colour trace, before POS**, where the artifact is still linear in the motion,
   - **from the fused pulse**, for what POS's per-window tuning made non-linear.

   It only uses pose channels that clearly move in a window (rotation ≥ 1°, translation ≥ 0.5 cm RMS). Building it showed why: MediaPipe's pose estimate jitters *with the pulse* (correlation up to 0.37 on still synthetic videos), so regressing on small pose changes subtracts real pulse.
8. **Breathing + heart rate:**
   - **Breathing rate** comes from the respiratory modulation of the pulse [10].
   - **Heart rate** comes from the FFT peak, rejecting peaks at the breathing rate and its harmonics. In irregular windows it is computed as 60 / mean beat interval instead, because the FFT peak is unreliable there.
9. **Beats and HRV:** beat detection with sub-frame peak refinement gives inter-beat intervals (IBIs), then **RMSSD** (beat-to-beat change; the core AF feature) and **SDNN** (overall spread; shown for information).
   - **LF/HF is dropped:** it needs 2–5 minute recordings, is distorted by slow breathing, its meaning is debated, and its main use was stress, which is no longer an output.
10. **AF screening (primary):**
    - **Features from the IBIs** over 30–60 s windows: normalised RMSSD (RMSSD ÷ mean IBI), Shannon and sample entropy, Poincaré SD1/SD2 and turning-point ratio, the standard set [20][21].
    - **Classifier:** a simple one, applied to **confident windows only**; otherwise the output is "signal unreliable".
    - **Extra beats** found by step 11 are excluded first, so they do not look like AF.
11. **Heart-rate flags (secondary):**
    - **slow / fast heart rate:** resting HR below 50 BPM (bradycardia [7]) or above 100 BPM (tachycardia), flagged only when it lasts at least 2 minutes in confident windows,
    - **extra (premature) beats:** an early beat followed by a longer pause, inside an otherwise regular rhythm [17].

---

## 2. What is new (and what is not)

**What already exists:**
- **AF from face video** has been shown before:
  - a smartphone camera on the face detected AF with 95% sensitivity and 96% specificity against a 12-lead ECG (Yan et al., 2018 [1]),
  - the same group did multi-person detection with deep learning (Yan et al., 2020 [2]),
  - deep learning on systolic peaks reached about 96% accuracy (Sun et al., 2022 [3]).
- **Weighting face regions by signal quality** also exists:
  - DistancePPG (Kumar et al., 2015 [18]),
  - skin-region selection (Bobbia et al. [12]),
  - orientation masking (Cantrill et al. [13]).

**The known weakness:** in rPPG, **head motion, talking and lighting changes make the beat intervals look irregular**. That is exactly what AF looks like, so motion produces **false AF alarms**.

**Our contribution** is a quality-aware pipeline built for rhythm screening:
1. **Per-patch trust that does not assume a regular rhythm** (3D orientation, brightness, cross-patch consistency, beat agreement), so noise is down-weighted but real AF is not.
2. **Head-pose motion removal** and **breathing-aware heart rate**, for cleaner beat timing.
3. **Confidence-gated rhythm screening:** AF is only judged in windows the system trusts; otherwise it abstains. We report the **false-alarm rate** and the **coverage** (how often the system gives an answer), with and without gating, on AF and non-AF recordings.

In our literature search we did not find this combination applied to AF screening; before submission we will repeat the search and state the claim as "to our knowledge".

---

## 3. Real-life use cases

### Primary: contactless atrial fibrillation screening
- **AF is common and often silent.** Prevalence is about 2–3% at ages 65–69 and about 10% over 85; many people with AF are unaware of it [5].
- **AF raises stroke risk up to 5-fold**, and about 20% of AF-related strokes occur in people whose AF is first found at the time of the stroke [5].
- **It is treatable once found:** most older adults with undiagnosed AF are eligible for stroke-preventing anticoagulation [5].
- **A camera-based check needs no wearable or electrodes:** it could run during telehealth calls or at a pharmacy or clinic kiosk, or as a periodic self-check at home.
- **Honest positioning:** the value of *mass* AF screening is still debated [5][6]. The output is a **screening aid** ("possible AF — see a doctor for an ECG"), **not a diagnosis**.

### Secondary: heart-rate flags
- **Slow heart rate (bradycardia):** a resting rate below 50 BPM [7]. Often normal in fit people or during sleep, but it can come from a failing natural pacemaker, a conduction block, medicines or an underactive thyroid. Warning signs are dizziness, fainting and tiredness; some people need a pacemaker.
- **Fast heart rate (tachycardia):** a resting rate above 100 BPM. Exercise, stress, caffeine and fever are normal causes; it can also come from an arrhythmia (AF itself is often fast), anaemia or an overactive thyroid.
- **Extra (premature) beats:** very common and usually harmless; checked by a doctor only when frequent. Detecting them also helps the primary output, because they are a known cause of false AF alarms [17].
- All three come from data we already compute, need no extra hardware, and are shown with context (e.g. "46 BPM: can be normal for fit people; see a doctor if you feel dizzy or faint"), as hints, not diagnoses.

---

## 4. Data

| Dataset | Contents | Used for | Access |
|---|---|---|---|
| **MIMIC PERform AF** [4] | contact PPG + ECG + respiration from 35 ICU patients (19 AF, 16 non-AF), 125 Hz, labelled | **build and validate the AF classifier**; test that the confidence measure does not reject AF windows | public (Zenodo) |
| **OBF** [11] | face videos with reference physiological signals from healthy subjects **and AF patients** (University of Oulu) | **test AF screening on real face videos** (the strongest possible result) | by request from the Oulu group; request early |
| **UBFC-rPPG**, **PURE** | face videos + PPG, normal rhythm; PURE has controlled head motion | HR / beat-timing accuracy; **false-alarm rate** of AF screening on normal rhythm, with vs without gating | open download / request |
| **UBFC-Phys** [8] | face videos + BVP, 56 subjects; rest, speech and arithmetic tasks | more normal-rhythm videos with talking (a hard case for false alarms); raised heart rates during the tasks | IEEE DataPort |
| **MIT-BIH Arrhythmia** [16] | ECG with every beat labelled (normal, premature atrial, premature ventricular, …) | build and test the extra-beat detector on beat timings | public (PhysioNet) |
| **COHFACE** [14] | face videos + PPG + respiration belt | breathing-rate accuracy | request from Idiap |
| MMPD (optional) | mobile face videos with skin tone, motion and lighting labels | robustness by condition | request |
| Synthetic videos | face videos with known beat timing: irregular (AF-like), slow (35–50 BPM), fast (100–150 BPM), with inserted extra beats; at several face sizes | end-to-end tests if OBF is unavailable; secondary flags; adaptive patch count | generated by us |

**If OBF access is not granted:** we cannot measure how often face video *catches* real AF (sensitivity). We can still show:
1. the AF classifier works on real AF pulse data (MIMIC PERform AF), and the confidence measure keeps AF windows,
2. our pipeline reduces **false alarms on face videos** of normal-rhythm subjects (UBFC-rPPG, PURE, UBFC-Phys), the main known weakness,
3. detection works end-to-end on synthetic irregular-rhythm videos.

---

## 5. Phases

### Phase 0: Setup and data access (week 1) — done (data requests are the team's)
- **Request OBF first** (decides the strength of the final result), then UBFC-Phys, COHFACE and PURE; download UBFC-rPPG, MIMIC PERform AF and MIT-BIH Arrhythmia.
- Reuse the existing environment setup and evaluation protocol (FFT HR, MAE/RMSE/MAPE/Pearson/SNR). Widen the heart-rate band from 0.7–3.0 Hz (42–180 BPM) to 0.5–3.3 Hz (30–200 BPM) so slow and fast rates can be measured at all.
- **Exit:** UBFC-rPPG, MIMIC PERform AF and MIT-BIH downloaded; data requests sent.

### Phase 1: Multi-patch core (weeks 2–3) — steps 1–7 — built; exit check waits for UBFC-rPPG / PURE
- Patch definitions on the face mesh at three levels (25 / 12 / 6), chosen per video from face size; per-patch colour traces, orientation and brightness; head pose.
- POS per patch; the four rhythm-independent trust checks; weighted fusion; confidence; motion removal.
- Tests on synthetic videos with known beat timing and deliberate artifacts (flicker, shine, head motion, small faces).
- **Exit:** the full method beats POS on a single skin region in HR MAE on UBFC-rPPG and PURE.

### Phase 2: Breathing, heart rate and beat timing (weeks 4–5) — steps 8–9
- Breathing rate from the pulse signal's respiratory modulation; breathing-aware HR selection; beat-based HR in irregular windows.
- Beat detection with sub-frame peak refinement; inter-beat intervals; RMSSD and SDNN.
- **Exit:**
  - breathing-rate error on COHFACE,
  - IBI and RMSSD/SDNN error vs ground-truth PPG on UBFC-rPPG (a published method reports RMSSD and SDNN errors of 10.5 ms and 6.15 ms [15]).

### Phase 3: AF screening (weeks 5–7) — step 10
- Irregularity features from IBIs over 30–60 s windows, after removing isolated extra beats (Phase 4); simple, explainable classifiers (logistic regression, random forest).
- Train and validate on MIMIC PERform AF with **subject-independent** cross-validation (sensitivity, specificity, AUC).
- Apply to face videos:
  - **OBF**, if available: sensitivity and specificity on real AF patients,
  - **UBFC-rPPG, PURE, UBFC-Phys**: false-alarm rate on normal rhythm,
  - each with vs without confidence gating, with coverage.
- **Exit:** a false-alarm rate on face videos that drops with gating, at acceptable coverage.

### Phase 4: Heart-rate flags, secondary (week 7) — step 11
- **Slow / fast:** thresholds (50 and 100 BPM) on the median resting HR over at least 2 minutes of confident windows. Evaluate as agreement between the flag from our HR and the flag from the reference sensor's HR on all face-video datasets, plus synthetic videos at 35–150 BPM (few real recordings are outside 50–100 BPM).
- **Extra beats:** rule on the IBI pattern (an IBI clearly shorter than the local median, followed by a longer one). Build and test on MIT-BIH beat timings with camera-like timing noise added (±33 ms at 30 fps), then on synthetic videos.
- **Exit:**
  - flag agreement on face videos,
  - extra-beat sensitivity and precision on MIT-BIH,
  - fewer AF false alarms on MIMIC PERform AF non-AF subjects once extra beats are removed.

### Phase 5: Evaluation and comparison (weeks 7–8)
- **Baselines (comparison only):** POS and CHROM on a single skin region; pretrained deep models from rPPG-Toolbox (PhysNet, TS-CAN, EfficientPhys, PhysFormer). Each feeds the same AF classifier, so the comparison isolates the pulse-extraction quality.
- **Ablation:** single region → patches → + each trust check → + motion removal → + breathing-aware HR → + confidence gating, measured on HR error, IBI error and AF false-alarm rate.
- **Patch count:** 6 vs 12 vs 25 patches vs adaptive, at several face sizes.
- **Confidence measure:** spectral-sharpness confidence vs our rhythm-independent confidence. Measure the share of AF windows each one rejects (on MIMIC PERform AF, and OBF if available). This shows why the design choice matters.
- **Motion removal check:** compare with vs without on still videos. If removal hurts there, it is subtracting the real head bobbing caused by each heartbeat [22].
- **Efficiency:** ms/frame on a laptop CPU.

### Phase 6: Live demo (weeks 8–9)
- Webcam: live HR, breathing rate, RMSSD, rhythm status, heart-rate flags and confidence, with the patch trust map drawn on the face.

### Phase 7: Report and presentation (weeks 9–10)

---

## 6. Evaluation metrics

| What | Metrics |
|---|---|
| Heart rate | MAE, RMSE, MAPE, Pearson r, SNR |
| Beat timing / HRV | IBI error (ms); absolute error of RMSSD and SDNN (ms) |
| Breathing rate | MAE vs respiration-belt reference (COHFACE) |
| **AF screening** | sensitivity, specificity, AUC (MIMIC PERform AF; OBF if available); **false-alarm rate on normal-rhythm face videos**; coverage with abstention, reported separately for AF and non-AF recordings |
| Heart-rate flags (secondary) | slow/fast: agreement with the flag from the reference HR (accuracy, Cohen's kappa); extra beats: sensitivity and precision (MIT-BIH, synthetic videos) |
| Efficiency | ms/frame on CPU |

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No access to face videos of AF patients (OBF) | Request on day 1; otherwise use the three-part evidence in Section 4 and state the limitation clearly |
| Domain gap: the classifier is trained on contact PPG (ICU), applied to camera pulses | Use timing-only features (IBIs), which transfer better than waveform shape; report results on face videos separately |
| AF needs accurate beat timing, which rPPG struggles with | Trust checks, motion removal, sub-frame peak refinement and gating; report IBI error and coverage |
| Quality checks that assume a regular rhythm would reject real AF | All trust and confidence measures are rhythm-independent; Phase 5 tests this directly |
| Small faces make 25 patches too noisy | Adaptive patch count (25 / 12 / 6); patch-count ablation |
| Motion removal could subtract real pulse (the head bobs slightly with each heartbeat [22]) | Ridge regression (a gentle fit); with/without comparison on still videos |
| MIMIC PERform AF is small (35 subjects) | Subject-independent cross-validation, simple models, confidence intervals |
| Breathing from the pulse signal is weaker than from a sensor | Validate on COHFACE; use breathing only as a helper and a feature |
| The wider heart-rate band (30–200 BPM) lets in more noise | Breathing-aware peak selection and confidence gating; slow/fast flags only after 2 minutes of confident windows |
| Few real face videos with slow or fast heart rates, and none labelled for extra beats | Agreement with the reference-sensor flag, synthetic videos, and MIT-BIH beat timings with camera-like noise; state the limitation |
| A slow or fast rate is often normal (fitness, exercise, caffeine) | Flag only at rest, show the reading with context, never as a diagnosis |
| Overclaiming novelty or medical use | "To our knowledge" novelty; screening aid, not a diagnosis |

---

## 8. Key references

- [1] Yan, B. P., et al. (2018). Contact-free screening of atrial fibrillation by a smartphone using facial pulsatile photoplethysmographic signals. *Journal of the American Heart Association*, 7(8), e008585.
- [2] Yan, B. P., et al. (2020). High-throughput, contact-free detection of atrial fibrillation from video with deep learning. *JAMA Cardiology*, 5(1), 105–107.
- [3] Sun, Z., et al. (2022). Non-contact atrial fibrillation detection from face videos by learning systolic peaks. *IEEE Journal of Biomedical and Health Informatics*. arXiv:2110.07610.
- [4] Charlton, P. H., et al. (2022). Detecting beats in the photoplethysmogram: benchmarking open-source algorithms. *Physiological Measurement*, 43(8), 085007. (MIMIC PERform AF dataset)
- [5] US Preventive Services Task Force (2018). Screening for atrial fibrillation with electrocardiography: final evidence summary.
- [6] Atrial fibrillation screening review. *Circulation Research* (2020). PMC7388078.
- [7] Kusumoto, F. M., et al. (2019). 2018 ACC/AHA/HRS Guideline on the Evaluation and Management of Patients With Bradycardia and Cardiac Conduction Delay. *Circulation*, 140(8), e382–e482.
- [8] Meziati Sabour, R., Benezeth, Y., De Oliveira, P., Chappé, J., & Yang, F. (2021). UBFC-Phys: A Multimodal Database for Psychophysiological Studies of Social Stress. *IEEE Transactions on Affective Computing*.
- [9] Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
- [10] Karlen, W., Raman, S., Ansermino, J. M., & Dumont, G. A. (2013). Multiparameter respiratory rate estimation from the photoplethysmogram. *IEEE Transactions on Biomedical Engineering*, 60(7), 1946–1953.
- [11] Li, X., et al. (2018). The OBF Database: A Large Face Video Database for Remote Physiological Signal Measurement and Atrial Fibrillation Detection. *IEEE FG 2018*, 242–249.
- [12] Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. *Pattern Recognition Letters*, 124, 82–90.
- [13] Cantrill, S., Ahmedt-Aristizabal, D., Petersson, L., Suominen, H., & Armin, M. A. (2024). Orientation-conditioned facial texture mapping for video-based facial remote photoplethysmography estimation. *CVPR Workshops*.
- [14] Heusch, G., Anjos, A., & Marcel, S. (2017). A reproducible study on remote heart rate measurement. arXiv:1709.00962. (COHFACE dataset)
- [15] Robust Heart Rate Variability Measurement from Facial Videos. PubMed Central, PMC10376629.
- [16] Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.
- [17] Chong, J. W., et al. (2015). Arrhythmia discrimination using a smart phone. *IEEE Journal of Biomedical and Health Informatics*, 19(3), 815–824.
- [18] Kumar, M., Veeraraghavan, A., & Sabharwal, A. (2015). DistancePPG: Robust non-contact vital signs monitoring using a camera. *Biomedical Optics Express*, 6(5), 1565–1588.
- [19] Orphanidou, C., et al. (2015). Signal-quality indices for the electrocardiogram and photoplethysmogram: derivation and applications to wireless monitoring. *IEEE Journal of Biomedical and Health Informatics*, 19(3), 832–838.
- [20] Dash, S., Chon, K. H., Lu, S., & Raeder, E. A. (2009). Automatic real time detection of atrial fibrillation. *Annals of Biomedical Engineering*, 37(9), 1701–1709.
- [21] McManus, D. D., et al. (2013). A novel application for the detection of an irregular pulse using an iPhone 4S in patients with atrial fibrillation. *Heart Rhythm*, 10(3), 315–319.
- [22] Balakrishnan, G., Durand, F., & Guttag, J. (2013). Detecting pulse from head motions in video. *IEEE CVPR*, 3430–3437.
