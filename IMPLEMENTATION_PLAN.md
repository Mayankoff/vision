# Implementation Plan — Quality-Aware Multi-Patch rPPG for Atrial Fibrillation Screening

BCSE417L Machine Vision · Mayank (23BAI0158), Vyomesh Pandita (23BAI0151)

> **Status: plan only.** Nothing in this plan is built yet; building starts when the team says so.
> Existing methods (POS, CHROM, PhysNet, …) are not part of the architecture; they are only
> **comparison baselines** in the evaluation. The system needs **one ordinary RGB camera**
> (webcam or phone); no extra hardware.

Architecture diagram: [`docs/figures/block_diagram_proposed.png`](docs/figures/block_diagram_proposed.png)

---

## 1. What we are building

A contactless system that takes **RGB video of a face** and outputs:

| Output | Role |
|---|---|
| **AF screening result**: "regular rhythm", "possible AF — see a doctor", or "signal unreliable" | **primary** real-life output (Section 3) |
| **Heart-rate abnormality flags**: "slow heart rate", "fast heart rate" (at rest), "extra beats" | **secondary** real-life output, reusing the same heart rate and beat timings (Section 3) |
| Heart rate (BPM), breathing rate, HRV (RMSSD, SDNN, LF/HF) | vital signs, and the inputs to the two analyses |
| Confidence | the system says when it cannot be trusted instead of giving a wrong answer |

### Pipeline (matches the diagram)

1. **Face landmarks + head pose:** MediaPipe FaceLandmarker gives 478 3D landmarks and a head rotation/translation per frame.
2. **Skin patches:** forehead and cheeks are split into about 25 patches defined on the face mesh, so they move with the face.
3. **Patch colour traces:** mean R, G, B of every patch in every frame.
4. **POS per patch:** 25 candidate pulse signals.
5. **Trust score per patch**, updated every few seconds, from four checks:
   - **pulse quality:** share of the patch pulse's spectral power in its sharpest peak,
   - **agreement:** weighted vote on the peak frequency across patches; disagreeing patches are down-weighted,
   - **face orientation:** patch surface normal from the 3D landmarks; patches turned away are down-weighted,
   - **brightness:** patches in shadow or with shine are down-weighted.
6. **Weighted fusion:** trust-weighted average of the patch pulses, plus a confidence value.
7. **Motion removal:** regress out the part of the fused pulse that follows head pose.
8. **Breathing + heart rate:** breathing rate from the respiratory modulation of the pulse [10]; heart rate from the FFT peak, rejecting peaks at the breathing rate and its harmonics.
9. **HRV:** beat detection gives inter-beat intervals (IBIs), then RMSSD, SDNN and LF/HF.
10. **AF screening (primary):** rhythm-irregularity features from the IBIs (e.g. RMSSD, sample/Shannon entropy, Poincaré plot spread, turning-point ratio) go to a simple classifier. It is applied to **confident windows only**; otherwise the output is "signal unreliable". Isolated extra beats found by step 11 are excluded first, so they do not look like AF.
11. **Heart-rate abnormalities (secondary):**
    - **slow / fast heart rate:** resting HR below 50 BPM (bradycardia [7]) or above 100 BPM (tachycardia), flagged only when it lasts at least 2 minutes in confident windows,
    - **extra (premature) beats:** an early beat followed by a longer pause, inside an otherwise regular rhythm [17].

---

## 2. What is new (and what is not)

**What already exists:** AF detection from face video has been shown before:
- a smartphone camera on the face detected AF with 95% sensitivity and 96% specificity against a 12-lead ECG (Yan et al., 2018 [1]),
- the same group did high-throughput, multi-person detection with deep learning (Yan et al., 2020 [2]),
- deep learning on systolic peaks from face video reached about 96% accuracy (Sun et al., 2022 [3]).

**The known weakness:** in rPPG, **head motion, talking and lighting changes make the beat intervals look irregular**. That is exactly what AF looks like, so motion produces **false AF alarms**.

**Our contribution** is a quality-aware pipeline that targets this weakness. Signal trust is measured **patch by patch** and carried all the way to the screening decision:
1. **Per-patch trust from four checks** (pulse quality, cross-patch agreement, 3D face orientation, brightness), plus motion removal using head pose, gives cleaner beat timing.
2. **Breathing-aware heart-rate selection.**
3. **Confidence-gated rhythm screening:** AF is only judged in windows the system trusts; otherwise it abstains. We report the **false-alarm rate** and **how often the system gives an answer (coverage)**, with and without gating.

The patch, trust and fusion ideas build on published work (POS [9]; skin-region selection, Bobbia et al. [12]; orientation masking, Cantrill et al. [13]). In our literature search we did not find this combination applied to AF screening; before submission we will repeat the search and state the claim as "to our knowledge".

---

## 3. Real-life use case: contactless atrial fibrillation screening

- **AF is common and often silent.** Prevalence is about 2–3% at ages 65–69 and about 10% over 85; many people with AF are unaware of it [5].
- **AF raises stroke risk up to 5-fold**, and about 20% of AF-related strokes occur in people whose AF is first found at the time of the stroke [5].
- **It is treatable once found:** most older adults with undiagnosed AF are eligible for stroke-preventing anticoagulation [5].
- **A camera-based check needs no wearable or electrodes:** it could run during telehealth calls or at a pharmacy or clinic kiosk, or as a periodic self-check at home.
- **Honest positioning:** the value of *mass* AF screening is still debated [5][6]. The output is a **screening aid** ("possible AF — see a doctor for an ECG"), **not a diagnosis**.

**Secondary output, heart-rate abnormalities:**
- **Slow heart rate (bradycardia):** a resting rate below 50 BPM [7]. Often normal in fit people or during sleep, but it can come from a failing natural pacemaker, a conduction block, medicines or an underactive thyroid. Warning signs are dizziness, fainting and tiredness; some people need a pacemaker.
- **Fast heart rate (tachycardia):** a resting rate above 100 BPM. Exercise, stress, caffeine and fever are normal causes; it can also come from an arrhythmia (AF itself is often fast), anaemia or an overactive thyroid.
- **Extra (premature) beats:** very common and usually harmless; checked by a doctor only when frequent. Detecting them also helps the primary output, because they are a known cause of false AF alarms [17].
- All three come from the heart rate and beat timings we already compute, so they need no extra hardware or training data. They are shown with context (e.g. "46 BPM: can be normal for fit people; see a doctor if you feel dizzy or faint"), as hints, not diagnoses.

---

## 4. Data

| Dataset | Contents | Used for | Access |
|---|---|---|---|
| **MIMIC PERform AF** [4] | contact PPG + ECG + respiration from 35 ICU patients (19 AF, 16 non-AF), 125 Hz, labelled | **build and validate the AF classifier** on real AF pulse data | public (Zenodo) |
| **OBF** [11] | face videos with reference physiological signals from healthy subjects **and AF patients** (University of Oulu) | **test AF screening on real face videos** (the strongest possible result) | by request from the Oulu group; request early |
| **UBFC-rPPG**, **PURE** | face videos + PPG, normal rhythm; PURE has controlled head motion | heart rate / HRV accuracy; **false-alarm rate** of AF screening on normal rhythm, with vs without gating | open download / request |
| **UBFC-Phys** [8] | face videos + BVP, 56 subjects; rest, speech and arithmetic tasks | more normal-rhythm videos with talking (a hard case for false alarms); raised heart rates during the tasks | IEEE DataPort |
| **MIT-BIH Arrhythmia** [16] | ECG with every beat labelled (normal, premature atrial, premature ventricular, ...) | build and test the extra-beat detector on beat timings | public (PhysioNet) |
| **COHFACE** [14] | face videos + PPG + respiration belt | breathing-rate accuracy | request from Idiap |
| MMPD (optional) | mobile face videos with skin tone, motion and lighting labels | robustness by condition | request |
| Synthetic videos | face videos with known beat timing: irregular (AF-like), slow (35–50 BPM), fast (100–150 BPM), with inserted extra beats | end-to-end test of AF detection if OBF is unavailable, and of the secondary flags | generated by us |

**If OBF access is not granted:** we cannot measure how often face video *catches* real AF (sensitivity). We can still show:
1. the AF classifier works on real AF pulse data (MIMIC PERform AF),
2. our pipeline reduces **false alarms on face videos** of normal-rhythm subjects (UBFC-rPPG, PURE, UBFC-Phys), the main known weakness,
3. detection works end-to-end on synthetic irregular-rhythm videos.

---

## 5. Phases

### Phase 0: Setup and data access (week 1)
- **Request OBF first** (decides the strength of the final result), then UBFC-Phys, COHFACE and PURE; download UBFC-rPPG, MIMIC PERform AF and MIT-BIH Arrhythmia.
- Reuse the existing environment setup and evaluation protocol (FFT HR, MAE/RMSE/MAPE/Pearson/SNR). Widen the heart-rate band from 0.7–3.0 Hz (42–180 BPM) to 0.5–3.3 Hz (30–200 BPM) so slow and fast rates can be measured at all.
- **Exit:** UBFC-rPPG and MIMIC PERform AF downloaded; data requests sent.

### Phase 1: Multi-patch core (weeks 2–3) — steps 1–7
- Patch definitions on the face mesh; per-patch colour traces; per-patch orientation; head pose.
- POS per patch; four trust checks; weighted fusion; motion removal; confidence.
- Tests on synthetic videos with known heart rates and deliberate artifacts (flicker, shine, head motion).
- **Exit:** the full method beats POS on a single skin region in HR MAE on UBFC-rPPG and PURE.

### Phase 2: Breathing, heart rate and beat timing (weeks 4–5) — steps 8–9
- Breathing rate from the pulse signal's respiratory modulation; breathing-aware HR selection.
- Beat detection and inter-beat intervals; HRV (RMSSD, SDNN, LF/HF).
- **Exit:**
  - breathing-rate error on COHFACE,
  - IBI and HRV error vs ground-truth PPG on UBFC-rPPG (a published method reports RMSSD and SDNN errors of 10.5 ms and 6.15 ms [15]).

### Phase 3: AF screening (weeks 5–7) — step 10
- Irregularity features from IBIs over 30–60 s windows, after removing isolated extra beats (Phase 4); simple, explainable classifiers (logistic regression, random forest).
- Train and validate on MIMIC PERform AF with **subject-independent** cross-validation (sensitivity, specificity, AUC).
- Apply to face videos:
  - **OBF**, if available: sensitivity and specificity on real AF patients,
  - **UBFC-rPPG, PURE, UBFC-Phys**: false-alarm rate on normal rhythm,
  - each with vs without confidence gating, with coverage.
- **Exit:** a false-alarm rate on face videos that drops with gating, at acceptable coverage.

### Phase 4: Heart-rate abnormalities, secondary (week 7) — step 11
- **Slow / fast flags:** thresholds (50 and 100 BPM) on the median resting HR over at least 2 minutes of confident windows. Evaluate as agreement between the flag from our HR and the flag from the reference sensor's HR, on all face-video datasets, plus synthetic videos at 35–150 BPM (few real recordings are outside 50–100 BPM).
- **Extra beats:** rule on the IBI pattern (an IBI clearly shorter than the local median followed by a longer one). Build and test on MIT-BIH Arrhythmia beat timings, with camera-like timing noise added (±33 ms at 30 fps), then on synthetic videos with inserted extra beats.
- **Exit:** flag agreement on face videos; extra-beat sensitivity and precision on MIT-BIH; fewer AF false alarms on MIMIC PERform AF non-AF subjects once extra beats are removed.

### Phase 5: Evaluation and comparison (weeks 7–8)
- **Baselines (comparison only):** POS and CHROM on a single skin region; pretrained deep models from rPPG-Toolbox (PhysNet, TS-CAN, EfficientPhys, PhysFormer). Each feeds the same AF classifier, so the comparison isolates the pulse-extraction quality.
- **Ablation:** single region → 25 patches → + each trust check → + motion removal → + breathing-aware HR → + confidence gating, measured on HR error, IBI error and AF false-alarm rate.
- **Efficiency:** ms/frame on a laptop CPU.

### Phase 6: Live demo (weeks 8–9)
- Webcam: live HR, breathing rate, HRV, rhythm status, heart-rate flags and confidence, with the patch trust map drawn on the face.

### Phase 7: Report and presentation (weeks 9–10)

---

## 6. Evaluation metrics

| What | Metrics |
|---|---|
| Heart rate | MAE, RMSE, MAPE, Pearson r, SNR |
| Beat timing / HRV | IBI error (ms); absolute error of RMSSD and SDNN (ms) |
| Breathing rate | MAE vs respiration-belt reference (COHFACE) |
| **AF screening** | sensitivity, specificity, AUC (MIMIC PERform AF; OBF if available); **false-alarm rate on normal-rhythm face videos**; coverage with abstention |
| Heart-rate flags (secondary) | slow/fast: agreement with the flag from the reference HR (accuracy, Cohen's kappa); extra beats: sensitivity and precision (MIT-BIH, synthetic videos) |
| Efficiency | ms/frame on CPU |

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No access to face videos of AF patients (OBF) | Request on day 1; otherwise use the three-part evidence in Section 4 and state the limitation clearly |
| Domain gap: the classifier is trained on contact PPG (ICU), applied to camera pulses | Use timing-only features (IBIs), which transfer better than waveform shape; report results on face videos separately |
| AF needs accurate beat timing, which rPPG struggles with | That is what the trust checks, motion removal and gating are for; report IBI error and coverage |
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
