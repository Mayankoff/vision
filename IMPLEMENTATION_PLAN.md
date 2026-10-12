# Implementation Plan — Quality-Aware Multi-Patch rPPG with Stress Estimation

BCSE417L Machine Vision · Mayank (23BAI0158), Vyomesh Pandita (23BAI0151)

> **Status: plan only.** Nothing in this plan is built yet; building starts when the team says so.
> This plan replaces the earlier benchmark-centred plan. Existing methods (POS, CHROM, PhysNet, …)
> are not part of the architecture; they are only **comparison baselines** in the evaluation.
> The system needs **one ordinary RGB camera** (webcam or phone); no extra hardware.

Architecture diagram: [`docs/figures/block_diagram_proposed.png`](docs/figures/block_diagram_proposed.png)

---

## 1. What we are building

A contactless system that takes **RGB video of a face** and outputs:

| Output | Why it matters |
|---|---|
| Heart rate (BPM) | core vital sign |
| Breathing rate (breaths/min) | second vital sign; also used to make the heart rate more reliable |
| Heart-rate variability (RMSSD, SDNN, LF/HF) | beat-to-beat timing; the physiological basis of the stress estimate |
| **Stress level** (low / moderate / high) | the real-life use case (Section 3) |
| Confidence / "signal unreliable" | the system says when it cannot be trusted instead of giving a wrong number |

### Pipeline (matches the diagram)

1. **Face landmarks + head pose:** MediaPipe FaceLandmarker gives 478 3D landmarks and a head rotation/translation per frame.
2. **Skin patches:** forehead and cheeks are split into about 25 patches defined on the face mesh, so they move with the face.
3. **Patch colour traces:** mean R, G, B of every patch in every frame.
4. **POS per patch:** 25 candidate pulse signals.
5. **Trust score per patch**, updated every few seconds, from four checks:
   - **pulse quality:** share of the patch pulse's spectral power in its sharpest peak,
   - **agreement:** weighted vote on the peak frequency across patches; disagreeing patches are down-weighted (protects against a patch with a rhythmic artifact such as flickering shine),
   - **face orientation:** patch surface normal from the 3D landmarks; patches turned away from the camera are down-weighted,
   - **brightness:** patches in shadow or with shine are down-weighted.
6. **Weighted fusion:** trust-weighted average of the patch pulses, plus a confidence value.
7. **Motion removal:** regress out the part of the fused pulse that follows head pose.
8. **Breathing + heart rate:** breathing rate from the respiratory modulation of the pulse signal (baseline, amplitude and frequency modulation [7]); heart rate from the FFT peak, **rejecting peaks at the breathing rate and its harmonics**, which are a common way rPPG locks onto the wrong frequency.
9. **HRV:** beat detection on the fused pulse gives inter-beat intervals, then RMSSD, SDNN and LF/HF. Only high-confidence windows are used.
10. **Stress estimation:** a classifier on HR, HRV and breathing rate gives low / moderate / high stress, or "unreliable" when confidence is low.

---

## 2. What is new (and what is not)

| Prior work | What it did | What it did **not** do |
|---|---|---|
| Wang et al., 2017 (POS) [8] | colour projection that turns RGB into a pulse | one region, equal weight for all pixels, no trust or confidence |
| Bobbia et al., 2019 [4] | unsupervised skin-region (superpixel) selection for rPPG | no motion removal, no 3D orientation, no stress |
| Cantrill et al., 2024 [9] | masks face regions by surface orientation (3D mesh) | used inside a deep network, no per-patch signal trust, no stress |
| Meziati Sabour et al., 2021 (UBFC-Phys) [1] | stress recognition from remote pulse-rate variability, 85.48% | single-region rPPG; no quality gating |
| Liu et al., 2024 [5] | rPPG features for cognitive stress | feature-level only; the pulse extraction itself is not quality-aware |

**Our contribution:** one pipeline in which **signal trust is measured patch by patch and carried all the way to the final answer**:
1. **Per-patch trust from four checks:** pulse quality, cross-patch agreement, 3D face orientation and brightness. Combined with motion removal using head pose.
2. **Breathing-aware heart-rate selection** using the breathing rate estimated from the same video.
3. **Confidence-gated stress estimation:** stress is only estimated from windows the system trusts, and it abstains otherwise. Performance is reported as accuracy against coverage.

In our literature search we did not find this combination. Before submission we will repeat the search to make sure the claim still holds, and state it as "to our knowledge".

---

## 3. Real-life use case: contactless stress / mental-workload monitoring

- **HRV is an established stress marker.** A review of 37 studies found that HRV changes under psychological stress, most often as reduced parasympathetic (high-frequency) activity [2].
- **Camera-based pulse can detect stress.** On UBFC-Phys, stress state recognition reached 85.48% accuracy using remote pulse-rate-variability features [1]. A systematic review and meta-analysis reports a pooled accuracy of about 85% for rPPG-based noncontact stress monitoring [3].
- **Breathing rate rises under stress** and complements HRV as a feature.
- **Applications:** workplace and student wellbeing, driver/operator fatigue and workload monitoring, telehealth check-ins, exam/interview stress studies. All need only a camera, no wearable.
- **Positioning:** the output is a **wellbeing / screening indicator, not a medical diagnosis**.

---

## 4. Data

| Dataset | Contents | Used for | Access |
|---|---|---|---|
| **UBFC-Phys** [1] | RGB face video, BVP + EDA wristband; 56 subjects; rest, speech and arithmetic tasks (Trier Social Stress Test style) | **main stress dataset**; published 85.48% to compare against | IEEE DataPort |
| **UBFC-rPPG** | RGB + PPG, 42 subjects, controlled | heart rate and HRV accuracy | open download |
| **COHFACE** | RGB + PPG + respiration belt, 40 subjects | breathing-rate accuracy (has real breathing ground truth) | request from Idiap |
| PURE / MMPD | RGB + PPG with head motion (PURE); mobile video with skin tone, motion and lighting labels (MMPD) | robustness of the heart rate under motion, lighting and skin tone (optional) | request |
| Own pilot recordings (optional) | webcam + a consumer pulse sensor or chest strap; rest vs mental arithmetic; self-reported stress (0–10) | live validation | needs informed consent from volunteers |

---

## 5. Phases

### Phase 0: Setup and data access (week 1)
- Request UBFC-Phys and COHFACE (plus PURE/MMPD if robustness analysis is wanted); download UBFC-rPPG.
- Reuse the existing environment setup and evaluation protocol (band 0.7–3.0 Hz, FFT HR, MAE/RMSE/MAPE/Pearson/SNR).
- **Exit:** UBFC-rPPG and UBFC-Phys downloaded and loaders verified.

### Phase 1: Multi-patch core (weeks 2–3) — steps 1–7
- Patch definitions on the face mesh (25 patches); per-patch colour traces; per-patch orientation; head pose.
- POS per patch; four trust checks; weighted fusion; motion removal; confidence.
- Tests on synthetic videos with known heart rates and deliberate artifacts (flicker, shine, head motion).
- **Exit:** the full method beats POS on a single skin region in HR MAE on UBFC-rPPG and PURE.

### Phase 2: Breathing and breathing-aware heart rate (week 4) — step 8
- Breathing rate from the pulse signal's respiratory modulation.
- Heart-rate peak selection that rejects breathing-rate peaks and their harmonics.
- **Exit:** breathing-rate error on COHFACE; HR error with vs without breathing-aware selection.

### Phase 3: HRV (week 5) — step 9
- Beat detection, inter-beat intervals, RMSSD, SDNN, LF/HF; confidence gating.
- **Exit:** HRV error vs ground-truth PPG on UBFC-rPPG. For reference, a published method reports RMSSD and SDNN errors of 10.5 ms and 6.15 ms on UBFC-rPPG [6].

### Phase 4: Stress estimation (weeks 6–7) — step 10
- Features per window: HR, HRV (RMSSD, SDNN, LF/HF), breathing rate, confidence.
- Simple, explainable classifiers (logistic regression, random forest); **leave-one-subject-out** cross-validation.
- UBFC-Phys: rest vs stress tasks; compare with the published 85.48%.
- Abstention: accuracy vs the fraction of windows answered (coverage), with and without confidence gating.

### Phase 5: Evaluation and comparison (weeks 7–8)
- **Baselines (comparison only, not part of the architecture):** POS and CHROM on a single skin region; pretrained deep models from rPPG-Toolbox (PhysNet, TS-CAN, EfficientPhys, PhysFormer).
- **Ablation:** single region → 25 patches → + trust checks (one at a time) → + motion removal → + breathing-aware HR; for stress: HRV only → + breathing rate → + confidence gating.
- **Efficiency:** ms/frame on a laptop CPU.

### Phase 6: Live demo (weeks 8–9)
- Webcam: live heart rate, breathing rate, HRV, stress level and confidence, with the patch trust map drawn on the face.

### Phase 7: Report and presentation (weeks 9–10)

---

## 6. Evaluation metrics

| What | Metrics |
|---|---|
| Heart rate | MAE, RMSE, MAPE, Pearson r, SNR |
| Breathing rate | MAE vs respiration-belt reference (COHFACE) |
| HRV | absolute error of RMSSD and SDNN (ms) |
| Stress | accuracy, F1, AUC (leave-one-subject-out); accuracy vs coverage with abstention |
| Efficiency | ms/frame on CPU |

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| HRV needs very clean pulses | Confidence gating; report coverage alongside accuracy |
| Breathing from the pulse signal is weaker than from a sensor | Validate on COHFACE; use breathing only as a helper and a feature, and report its error |
| UBFC-Phys stress labels are task-based (rest vs speech/arithmetic), not per-person feelings | State this clearly; optional pilot recordings with self-reports |
| Small subject numbers for stress | Leave-one-subject-out validation, simple models, confidence intervals |
| Dataset access delays | Request on day 1; build and test on UBFC-rPPG and synthetic videos meanwhile |
| Overclaiming novelty or medical use | "To our knowledge" novelty; wellbeing indicator, not a diagnosis |

---

## 8. Key references

- [1] Meziati Sabour, R., Benezeth, Y., De Oliveira, P., Chappé, J., & Yang, F. (2021). UBFC-Phys: A Multimodal Database for Psychophysiological Studies of Social Stress. *IEEE Transactions on Affective Computing*.
- [2] Kim, H.-G., Cheon, E.-J., Bai, D.-S., Lee, Y. H., & Koo, B.-H. (2018). Stress and Heart Rate Variability: A Meta-Analysis and Review of the Literature. *Psychiatry Investigation*, 15(3), 235–245.
- [3] Performance of artificial intelligence models for monitoring psychological stress using noncontact physiological signals: a systematic review and meta-analysis. PubMed Central, PMC13534124. https://pmc.ncbi.nlm.nih.gov/articles/PMC13534124/
- [4] Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. *Pattern Recognition Letters*, 124, 82–90.
- [5] Liu, I., et al. (2024). Your blush gives you away: detecting hidden mental states with remote photoplethysmography and thermal imaging. *PeerJ Computer Science*, 10, e1912.
- [6] Robust Heart Rate Variability Measurement from Facial Videos. PubMed Central, PMC10376629. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10376629/
- [7] Karlen, W., Raman, S., Ansermino, J. M., & Dumont, G. A. (2013). Multiparameter respiratory rate estimation from the photoplethysmogram. *IEEE Transactions on Biomedical Engineering*, 60(7), 1946–1953.
- [8] Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
- [9] Cantrill, S., Ahmedt-Aristizabal, D., Petersson, L., Suominen, H., & Armin, M. A. (2024). Orientation-conditioned facial texture mapping for video-based facial remote photoplethysmography estimation. *CVPR Workshops*.
- [10] Heusch, G., Anjos, A., & Marcel, S. (2017). A reproducible study on remote heart rate measurement. arXiv:1709.00962. (COHFACE dataset)
