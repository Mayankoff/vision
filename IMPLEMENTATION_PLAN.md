# Implementation Plan — Thermal-Guided, Quality-Aware Multi-Patch rPPG with Stress Estimation

BCSE417L Machine Vision · Mayank (23BAI0158), Vyomesh Pandita (23BAI0151)

> **Status: plan only.** Nothing in this plan is built yet; building starts when the team says so.
> This plan replaces the earlier benchmark-centred plan. Existing methods (POS, CHROM, PhysNet, …)
> are no longer part of the architecture. They are only **comparison baselines** in the evaluation.

Architecture diagram: [`docs/figures/block_diagram_proposed.png`](docs/figures/block_diagram_proposed.png)

---

## 1. What we are building

A contactless system that takes **synchronised RGB + thermal video of a face** and outputs:

| Output | Why it matters |
|---|---|
| Heart rate (BPM) | core vital sign |
| Breathing rate (breaths/min) | second vital sign, from thermal; also used to make the heart rate more reliable |
| Heart-rate variability (RMSSD, SDNN, LF/HF) | beat-to-beat timing; the physiological basis of the stress estimate |
| **Stress level** (low / moderate / high) | the real-life use case (Section 3) |
| Confidence / "signal unreliable" | the system says when it cannot be trusted instead of giving a wrong number |

### Pipeline (matches the diagram)

**RGB stream**
1. **Face landmarks + head pose:** MediaPipe FaceLandmarker gives 478 3D landmarks and a head rotation/translation per frame.
2. **Skin patches:** forehead and cheeks are split into about 25 patches defined on the face mesh, so they move with the face.
3. **Patch colour traces:** mean R, G, B of every patch in every frame.
4. **POS per patch:** 25 candidate pulse signals.

**Thermal stream**
- **T1. RGB-thermal registration:** map landmarks and patches from the RGB frame onto the thermal frame.
- **T2. Thermal ROIs:** temperature of each patch, the nose tip and the nostril region.
- **T3. Nose-tip temperature:** slow trend; the nose tip cools under stress (sympathetic vasoconstriction).
- **T4. Breathing signal:** nostril temperature rises and falls with each breath, giving the breathing rate (RR).

**Trust, fusion and analysis**
5. **Trust score per patch**, updated every few seconds, from five checks:
   - **pulse quality:** share of the patch pulse's spectral power in its sharpest peak,
   - **agreement:** weighted vote on the peak frequency across patches; disagreeing patches are down-weighted,
   - **face orientation:** patch surface normal from the 3D landmarks; patches turned away are down-weighted,
   - **brightness:** patches in shadow or with shine are down-weighted,
   - **thermal skin check (new):** a patch that is not at skin temperature (hair, glasses, occlusion, background) is down-weighted. Thermal is unaffected by visible light, so a sudden RGB change with no thermal change also flags a lighting artifact.
6. **Weighted fusion:** trust-weighted average of the patch pulses, plus a confidence value.
7. **Motion removal:** regress out the part of the fused pulse that follows head pose.
8. **Heart + breathing rate (breathing-aware):** FFT peak picking that **rejects peaks at the breathing rate and its harmonics**. These are a common way rPPG locks onto the wrong frequency, and thermal gives us the breathing rate to avoid them.
9. **HRV:** beat detection on the fused pulse gives inter-beat intervals, then RMSSD, SDNN and LF/HF. Only high-confidence windows are used.
10. **Stress estimation:** a classifier on HR, HRV, RR and the nose-tip temperature trend gives low / moderate / high stress, or "unreliable" when confidence is low.

---

## 2. What is new (and what is not)

Honest positioning against the closest prior work:

| Prior work | What it did | What it did **not** do |
|---|---|---|
| Negishi et al., 2020 (Sensors) | RGB-thermal camera: HR from RGB, RR from nose/mouth temperature, body temperature; influenza screening | no multi-patch trust, no thermal-guided pulse extraction, no stress |
| Cho et al., 2019 (JMIR Mental Health) | smartphone (contact) PPG + thermal nose tip for perceived stress | contact PPG, not camera rPPG |
| Liu et al., 2024 (PeerJ CS) | fuses rPPG features + facial thermal features for cognitive stress | feature-level fusion only; thermal is not used to improve the pulse signal itself |
| iBVP dataset paper, Joshi & Cho, 2024 | RGB + thermal rPPG dataset; tests BVP from thermal frames | no patch-level trust or breathing-aware HR |

**Our contribution:** thermal is used **inside** the pulse extraction, not just as an extra feature:
1. **Thermal skin validity as a per-patch trust signal** in a quality-aware multi-patch rPPG.
2. **Breathing-aware heart-rate selection** using the thermal breathing rate.
3. **Confidence-gated stress estimation:** stress is only estimated from windows the system trusts, and it abstains otherwise.

In our literature search we did not find this combination. Before submission we will repeat the search to make sure the claim still holds, and state it as "to our knowledge".

---

## 3. Real-life use case: contactless stress / mental-workload monitoring

**Why stress:**
- **HRV is an established stress marker.** A review of 37 studies found that HRV changes under psychological stress, most often as reduced parasympathetic (high-frequency) activity (Kim et al., 2018, *Psychiatry Investigation*).
- **Remote PPG can detect stress.** On UBFC-Phys, stress state recognition reached 85.48% accuracy using remote pulse-rate-variability features (Meziati Sabour et al., 2021, *IEEE Trans. Affective Computing*). A recent systematic review and meta-analysis reports pooled accuracies of about 85% for rPPG and 73% for thermal imaging in noncontact stress monitoring [10].
- **Facial temperature tracks arousal and stress.** Nose-tip temperature follows emotional arousal (Kosonogov et al., 2017, *PLOS ONE*), and stress tasks lower nose-tip temperature (reviewed in Ioannou et al., 2014, *Psychophysiology*).
- **Applications:** workplace and student wellbeing, driver/operator fatigue and workload monitoring, telehealth check-ins, and exam/interview stress studies. All need no wearable.

**Secondary use case (described, not evaluated):** fever / infection screening by combining HR, RR and skin temperature, as in Negishi et al., 2020. No public dataset exists for it, so we only discuss it.

**Positioning:** the output is a **wellbeing / screening indicator, not a medical diagnosis**.

---

## 4. Data

| Dataset | Modalities | Used for | Access |
|---|---|---|---|
| **iBVP** (Joshi & Cho, 2024) | RGB + thermal face video, ear PPG ground truth with signal-quality labels; 31 subjects, 124 videos of ~3 min; conditions: controlled breathing, math tasks, head movement | main dataset: HR, HRV and RR accuracy; effect of the thermal modules | public for research (request) |
| **UBFC-Phys** (Meziati Sabour et al., 2021) | RGB face video, BVP + EDA wristband; 56 subjects; rest, speech and arithmetic tasks (Trier Social Stress Test style) | stress classification (RGB part); published 85.48% to compare against | IEEE DataPort |
| UBFC-rPPG | RGB + PPG, controlled | sanity checks of the RGB stream | open download |
| **Own pilot recordings** (optional) | webcam + FLIR Lepton 3.5 thermal (160x120, 8.7 Hz, ~US$370 with USB board), consumer pulse sensor or chest strap; rest vs mental arithmetic, self-reported stress (0–10) | the only way to evaluate stress **with** thermal features | needs informed consent from volunteers |

**To verify when the data arrives:**
- iBVP's thermal resolution and frame rate, and whether its RGB and thermal frames are already aligned.
- Whether iBVP's math tasks have difficulty levels, and whether its breathing condition is paced at a known rate (which would give a breathing-rate reference). iBVP has no respiration ground truth; a reference can also be derived from the PPG's respiratory modulation.
- Paced breathing changes HRV strongly, so iBVP's breathing condition **cannot** serve as a "low stress" class.

---

## 5. Phases

### Phase 0: Setup and data access (week 1)
- Request iBVP and UBFC-Phys; download UBFC-rPPG.
- Reuse the existing environment setup and evaluation protocol (band 0.7–3.0 Hz, FFT HR, MAE/RMSE/MAPE/Pearson/SNR).
- Decide on the pilot recordings: buy or borrow a thermal camera, or not.
- **Exit:** at least iBVP and UBFC-Phys downloaded and loaders verified.

### Phase 1: RGB multi-patch core (weeks 2–3) — steps 1–7
- Patch definitions on the face mesh (25 patches); per-patch colour traces; per-patch orientation; head pose.
- POS per patch; trust checks (pulse quality, agreement, orientation, brightness); weighted fusion; motion removal; confidence.
- Tests on synthetic videos with known heart rates and deliberate artifacts (flicker, shine, head motion).
- **Exit:** on iBVP and UBFC-rPPG, the full RGB method beats POS on a single skin region in HR MAE.

### Phase 2: Thermal stream (weeks 3–5) — T1–T4 and the thermal skin check
- RGB-thermal registration: use the dataset alignment if available, otherwise estimate an affine/homography transform per video (gradient-based image alignment, or a one-off calibration for our own camera rig).
- Thermal ROIs: patch temperatures, nose tip, nostrils.
- Thermal skin check added to the trust score; RGB-vs-thermal illumination-artifact check.
- Breathing signal from the nostril ROI, giving RR; breathing-aware HR peak selection.
- Nose-tip temperature trend.
- **Exit:** on iBVP, thermal modules on vs off; HR error decreases, and RR agrees with the available reference.

### Phase 3: HRV (week 5)
- Beat detection, inter-beat intervals, RMSSD, SDNN, LF/HF; confidence gating.
- **Exit:** HRV error vs ground-truth PPG on iBVP and UBFC-rPPG (for reference, a published RGB method reports RMSSD and SDNN errors of 10.5 ms and 6.15 ms on UBFC-rPPG [11]).

### Phase 4: Stress estimation (weeks 6–7) — step 10
- Features per window: HR, HRV (RMSSD, SDNN, LF/HF), RR, nose-tip temperature slope, confidence.
- Simple, explainable classifiers (logistic regression, random forest); **leave-one-subject-out** cross-validation.
- UBFC-Phys: rest vs stress tasks, RGB features only; compare with the published 85.48%.
- Pilot recordings (if collected): RGB-only vs RGB + thermal features.
- Abstention: accuracy vs the fraction of windows answered (coverage), with and without confidence gating.

### Phase 5: Evaluation and comparison (weeks 7–8)
- **Baselines (comparison only, not part of the architecture):** POS and CHROM on a single skin region; pretrained deep models from rPPG-Toolbox (PhysNet, TS-CAN, EfficientPhys, PhysFormer).
- **Ablation:** RGB patches → + trust checks → + motion removal → + thermal skin check → + breathing-aware HR; for stress: HRV only → + RR → + nose temperature → + confidence gating.
- **Efficiency:** ms/frame on a laptop CPU.

### Phase 6: Live demo (weeks 8–9)
- Webcam, plus a thermal camera if available: live HR, RR, HRV, stress level and confidence, with the patch trust map drawn on the face.

### Phase 7: Report and presentation (weeks 9–10)

---

## 6. Evaluation metrics

| What | Metrics |
|---|---|
| Heart rate | MAE, RMSE, MAPE, Pearson r, SNR |
| Breathing rate | MAE vs reference |
| HRV | absolute error of RMSSD and SDNN (ms) |
| Stress | accuracy, F1, AUC (leave-one-subject-out); accuracy vs coverage with abstention |
| Efficiency | ms/frame on CPU |

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No public dataset has RGB + thermal **and** stress labels | Stress validated on UBFC-Phys (RGB); thermal contribution to stress shown on pilot recordings; thermal contribution to HR/RR shown on iBVP |
| Low thermal frame rate (~9 Hz on cheap cameras) | Thermal is used for breathing, temperature and trust, never for the pulse itself |
| RGB-thermal registration errors | Check alignment visually; use large thermal ROIs; per-video alignment |
| HRV needs very clean pulses | Confidence gating; report coverage alongside accuracy |
| Small subject numbers for stress | Leave-one-subject-out validation, simple models, report confidence intervals |
| Thermal camera cost | Pilot recordings are optional; the core results use public datasets |
| Overclaiming novelty or medical use | Position it as "to our knowledge" novel and as a wellbeing indicator, not a diagnosis |

---

## 8. Key references

- [1] Joshi, J., & Cho, Y. (2024). iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels. *Electronics*, 13(7), 1334.
- [2] Negishi, T., et al. (2020). Contactless Vital Signs Measurement System Using RGB-Thermal Image Sensors and Its Clinical Screening Test on Patients with Seasonal Influenza. *Sensors*, 20(8), 2171.
- [3] Cho, Y., Julier, S. J., & Bianchi-Berthouze, N. (2019). Instant Stress: Detection of Perceived Mental Stress Through Smartphone Photoplethysmography and Thermal Imaging. *JMIR Mental Health*, 6(4), e10140.
- [4] Liu, I., et al. (2024). Your blush gives you away: detecting hidden mental states with remote photoplethysmography and thermal imaging. *PeerJ Computer Science*, 10, e1912.
- [5] Meziati Sabour, R., Benezeth, Y., De Oliveira, P., Chappé, J., & Yang, F. (2021). UBFC-Phys: A Multimodal Database for Psychophysiological Studies of Social Stress. *IEEE Transactions on Affective Computing*.
- [6] Kim, H.-G., Cheon, E.-J., Bai, D.-S., Lee, Y. H., & Koo, B.-H. (2018). Stress and Heart Rate Variability: A Meta-Analysis and Review of the Literature. *Psychiatry Investigation*, 15(3), 235–245.
- [7] Kosonogov, V., et al. (2017). Facial thermal variations: A new marker of emotional arousal. *PLOS ONE*, 12(9), e0183592.
- [8] Ioannou, S., Gallese, V., & Merla, A. (2014). Thermal infrared imaging in psychophysiology: Potentialities and limits. *Psychophysiology*, 51(10), 951–963.
- [9] Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491. (POS)
- [10] Performance of artificial intelligence models for monitoring psychological stress using noncontact physiological signals: a systematic review and meta-analysis. PubMed Central, PMC13534124. https://pmc.ncbi.nlm.nih.gov/articles/PMC13534124/
- [11] Robust Heart Rate Variability Measurement from Facial Videos. PubMed Central, PMC10376629. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10376629/
