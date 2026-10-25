"""Phase 4 experiments.

Extra beats (MIT-BIH Arrhythmia, beat annotations):
    tune the rule on DS1, test on DS2 (de Chazal et al., 2004), with camera-like beat-timing
    noise (0, 16 ms = measured on UBFC-rPPG, 33 ms = one frame at 30 fps). Scored outside AF /
    flutter episodes, since the rule is for an otherwise regular rhythm. Inside AF, the share of
    beats the rule flags per 30 s window shows whether ``max_share`` keeps AF from being "cleaned".

Slow / fast heart rate (MIMIC PERform AF): flag from the camera-like PPG beats vs the flag from
the ECG R-peaks, per 2-minute segment.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from afscreen.flags.ectopy import EctopyRule, detect_extra_beats
from afscreen.flags.mitbih import DS1, DS2, IGNORE, PREMATURE, BeatRecord

JITTERS_MS = (0.0, 16.0, 33.0)
GRID = {"prematurity": (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25),
        "recovery": (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)}
# Removal settings tried on DS1: (max_share, neighbour_tol); max_share 1.0 = no cap, tol None = no check.
REMOVAL = tuple((c, t) for c in (0.2, 1.0) for t in (None, 0.08, 0.10, 0.15, 0.20))
MAX_AF_LOSS = 0.02  # chosen setting may lose at most this much AF sensitivity on DS1


def removal_name(cap: float, tol: float | None) -> str:
    return f"cap{cap:g}" + ("" if tol is None else f"_tol{tol:g}")


def jittered(times: np.ndarray, jitter_ms: float, seed: int) -> np.ndarray:
    if not jitter_ms:
        return times
    return times + np.random.default_rng(seed).normal(0, jitter_ms / 1000, len(times))


def score_record(rec: BeatRecord, rule: EctopyRule, jitter_ms: float = 0.0, seed: int = 0) -> dict:
    """TP / FP / FN of the rule outside AF; SVEB and VEB recall separately."""
    hits = np.zeros(len(rec.times), dtype=bool)
    hits[detect_extra_beats(jittered(rec.times, jitter_ms, seed), rule)] = True
    truth = np.isin(rec.symbols, list(PREMATURE))
    scored = ~rec.in_af & ~np.isin(rec.symbols, list(IGNORE))
    sv, v = np.isin(rec.symbols, list("AaJS")), rec.symbols == "V"
    return {"record": rec.id, "beats": int(scored.sum()), "premature": int((truth & scored).sum()),
            "tp": int((hits & truth & scored).sum()), "fp": int((hits & ~truth & scored).sum()),
            "fn": int((~hits & truth & scored).sum()),
            "tp_sveb": int((hits & sv & scored).sum()), "n_sveb": int((sv & scored).sum()),
            "tp_veb": int((hits & v & scored).sum()), "n_veb": int((v & scored).sum())}


def pooled(rows: list[dict]) -> dict:
    d = pd.DataFrame(rows).sum(numeric_only=True)
    sens = d.tp / max(d.tp + d.fn, 1)
    prec = d.tp / max(d.tp + d.fp, 1)
    return {"sensitivity": sens, "precision": prec, "F1": 2 * sens * prec / max(sens + prec, 1e-12),
            "sveb_sens": d.tp_sveb / max(d.n_sveb, 1), "veb_sens": d.tp_veb / max(d.n_veb, 1),
            "false_per_1000_normal": 1000 * d.fp / max(d.beats - d.premature, 1)}


def tune(recs: dict[str, BeatRecord], jitter_ms: float = 16.0) -> tuple[EctopyRule, pd.DataFrame]:
    """Grid search on DS1 for the best pooled F1 at the given timing noise."""
    rows = []
    for p, r in product(GRID["prematurity"], GRID["recovery"]):
        rule = EctopyRule(prematurity=p, recovery=r)
        res = pooled([score_record(recs[k], rule, jitter_ms, i) for i, k in enumerate(DS1) if k in recs])
        rows.append({"prematurity": p, "recovery": r, **res})
    grid = pd.DataFrame(rows).sort_values("F1", ascending=False)
    best = grid.iloc[0]
    return EctopyRule(prematurity=float(best.prematurity), recovery=float(best.recovery)), grid


def flagged_share_by_window(recs: dict[str, BeatRecord], rule: EctopyRule, keys, jitter_ms: float = 16.0,
                            win_s: float = 30.0) -> pd.DataFrame:
    """Share of beats flagged as extra in each 30 s window: AF windows vs non-AF windows."""
    rows = []
    for i, k in enumerate(keys):
        rec = recs.get(k)
        if rec is None:
            continue
        t = jittered(rec.times, jitter_ms, i)
        for a in np.arange(0, t[-1] - win_s, win_s):
            m = (t >= a) & (t < a + win_s)
            if m.sum() < 12:
                continue
            af = rec.in_af[m]
            if 0 < af.mean() < 1:  # mixed window: skip
                continue
            n_extra = len(detect_extra_beats(t[m], rule))
            rows.append({"record": k, "af": bool(af[0]), "beats": int(m.sum()),
                         "has_premature": bool(np.isin(rec.symbols[m], list(PREMATURE)).any()),
                         "flagged_share": n_extra / m.sum()})
    return pd.DataFrame(rows)


def af_windows_with_removal(recs: dict[str, BeatRecord], keys, bundle: dict, rule: EctopyRule,
                            settings=REMOVAL, jitter_ms: float = 16.0, win_s: float = 30.0) -> pd.DataFrame:
    """AF probability of every 30 s MIT-BIH window, without and with extra-beat removal per cap.

    A second, independent test of the MIMIC-trained classifier (ECG beat timing instead of PPG),
    and the direct test of Phase 4's purpose: removal should lower the AF probability of
    non-AF windows with extra beats without lowering it for AF windows.
    """
    from dataclasses import replace

    from afscreen.af.features import ibi_features
    from afscreen.flags.ectopy import ibis_without_extra_beats

    model = bundle["model"]
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1  # many small predictions: worker start-up would dominate
    rows, feats = [], []  # feats: (row index, column, feature vector), predicted in one batch at the end
    for i, k in enumerate(keys):
        rec = recs.get(k)
        if rec is None:
            continue
        t = jittered(rec.times, jitter_ms, 1000 + i)
        for a in np.arange(0, t[-1] - win_s, win_s):
            m = (t >= a) & (t < a + win_s)
            af = rec.in_af[m]
            if m.sum() < 12 or 0 < af.mean() < 1:
                continue
            b = t[m]
            f0 = ibi_features(np.diff(b))
            if f0 is None:
                continue
            row = {"record": k, "af": bool(af[0]),
                   "has_premature": bool(np.isin(rec.symbols[m], list(PREMATURE)).any())}
            feats.append((len(rows), "p_none", f0))
            for c, tol in settings:
                ibis, _, _ = ibis_without_extra_beats(b, replace(rule, max_share=c, neighbour_tol=tol))
                f = ibi_features(ibis)
                feats.append((len(rows), f"p_{removal_name(c, tol)}", f0 if f is None else f))
            rows.append(row)
    if feats:
        p = model.predict_proba(np.array([f for _, _, f in feats]))[:, 1]
        for (r, col, _), pi in zip(feats, p):
            rows[r][col] = float(pi)
    return pd.DataFrame(rows)


def removal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of windows called AF (p >= 0.5) per group (rows), without and with each removal setting."""
    group = np.where(df.af, "AF", np.where(df.has_premature, "non-AF with extra beats", "non-AF without"))
    cols = [c for c in df.columns if c.startswith("p_")]
    t = df.assign(group=group).groupby("group")[cols].agg(lambda p: float((p >= 0.5).mean()))
    t.columns = [c[2:] for c in cols]
    t.insert(0, "windows", df.assign(group=group).groupby("group").size())
    return t


def choose_removal(ds1: pd.DataFrame, max_af_loss: float = MAX_AF_LOSS) -> str:
    """Setting with the fewest false alarms on non-AF windows with extra beats, among those that
    lose at most ``max_af_loss`` of AF sensitivity versus no removal (decided on DS1 only)."""
    base = ds1.loc["AF", "none"]
    cand = [c for c in ds1.columns if c not in ("windows", "none") and ds1.loc["AF", c] >= base - max_af_loss]
    if not cand:
        return "none"
    return min(cand, key=lambda c: (ds1.loc["non-AF with extra beats", c], -ds1.loc["AF", c]))


def run_ectopy(root: str | Path, out_dir: str | Path,
               model_path: str | Path = "results/phase3/mimic/af_classifier.joblib") -> dict:
    from afscreen.flags.mitbih import load_mitbih

    recs = load_mitbih(root)
    if not recs:
        raise SystemExit(f"no MIT-BIH records under {root}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rule, grid = tune(recs)
    grid.to_csv(out / "tuning_ds1.csv", index=False)
    test = []
    for j in JITTERS_MS:
        rows = [score_record(recs[k], rule, j, i) for i, k in enumerate(DS2) if k in recs]
        pd.DataFrame(rows).to_csv(out / f"per_record_ds2_jitter{int(j)}ms.csv", index=False)
        test.append({"jitter_ms": j, **pooled(rows)})
    test = pd.DataFrame(test)
    test.to_csv(out / "test_ds2.csv", index=False)
    shares = flagged_share_by_window(recs, rule, list(recs))
    shares.to_csv(out / "flagged_share_windows.csv", index=False)
    group = np.where(shares.af, "AF", np.where(shares.has_premature, "non-AF with extra beats", "non-AF without"))
    by = shares.groupby(group).flagged_share
    share_tbl = by.describe(percentiles=[0.1, 0.5, 0.9]).assign(
        above_max_share=by.apply(lambda s: float((s > rule.max_share).mean())))
    share_tbl.to_csv(out / "flagged_share_summary.csv")

    import joblib

    bundle = joblib.load(model_path)
    removal = {}
    for split, keys in (("DS1", DS1), ("DS2", DS2)):
        w = af_windows_with_removal(recs, keys, bundle, rule)
        w.to_csv(out / f"af_windows_{split}.csv", index=False)
        removal[split] = removal_table(w)
        removal[split].to_csv(out / f"af_removal_{split}.csv")
    chosen = choose_removal(removal["DS1"])
    (out / "chosen_removal.txt").write_text(chosen + "\n")
    (out / "summary.md").write_text(_ectopy_md(rule, grid, test, share_tbl, len(recs), removal, chosen),
                                    encoding="utf-8")
    return {"rule": rule, "test": test, "shares": share_tbl, "removal": removal, "chosen": chosen}


def _md(df: pd.DataFrame, index: bool = False) -> str:
    df = df.reset_index() if index else df
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(lines)


def _ectopy_md(rule, grid, test, shares, n, removal, chosen) -> str:
    rem = []
    for split, t in removal.items():
        rem += [f"### {split}\n", _md(t, index=True), ""]
    d2 = removal["DS2"]
    rem += [f"**Chosen on DS1** (fewest false alarms with at most {MAX_AF_LOSS:.0%} AF sensitivity lost): `{chosen}`. "
            f"DS2: AF {d2.loc['AF', 'none']:.2f} -> {d2.loc['AF', chosen]:.2f}; non-AF with extra beats "
            f"{d2.loc['non-AF with extra beats', 'none']:.2f} -> {d2.loc['non-AF with extra beats', chosen]:.2f}.\n"]
    return "\n".join([
        "# Phase 4: extra-beat detection on MIT-BIH Arrhythmia\n",
        f"{n} records (paced 102, 104, 107, 217 excluded). Rule tuned on DS1 (best F1 at 16 ms timing noise), "
        "tested on DS2; scored outside AF / flutter; fusion and unclassifiable beats ignored.\n",
        f"**Rule:** IBI before < (1 - {rule.prematurity:.2f}) x local median, IBI after > {rule.recovery:.2f} x local median "
        f"(median of {rule.context} intervals each side).\n",
        "## DS2 test\n", _md(test), "",
        "## Top of the DS1 grid\n", _md(grid.head(5)), "",
        f"## Share of beats flagged per 30 s window (removal is skipped above {rule.max_share:.2f})\n",
        _md(shares, index=True), "",
        "## AF classifier (trained on MIMIC) on MIT-BIH 30 s windows: share called AF\n",
        "Beat times from the ECG annotations with 16 ms timing noise. `none` = no removal; `capX` = extra beats "
        "removed unless more than X of the beats are flagged; `_tolY` = only extra beats whose neighbouring "
        "intervals are within Y of the local median. For AF windows this is sensitivity; for the others, the "
        "false-alarm rate.\n", *rem,
    ]) + "\n"


def chosen_rule(mitbih_dir: str | Path = "results/phase4/mitbih") -> EctopyRule:
    """The tuned detection rule plus the removal setting chosen on DS1 (from run_ectopy's outputs)."""
    d = Path(mitbih_dir)
    best = pd.read_csv(d / "tuning_ds1.csv").iloc[0]
    name = (d / "chosen_removal.txt").read_text().strip()
    cap, tol = 1.0, None
    for part in name.split("_"):
        if part.startswith("cap"):
            cap = float(part[3:])
        elif part.startswith("tol"):
            tol = float(part[3:])
    return EctopyRule(prematurity=float(best.prematurity), recovery=float(best.recovery), max_share=cap,
                      neighbour_tol=tol)


def run_mimic_removal(mimic_root: str | Path, out_dir: str | Path, rule: EctopyRule, camera_fs: float = 30.0,
                      jitter_ms: float = 16.0) -> pd.DataFrame:
    """MIMIC PERform AF, LOSO, camera-like beats: AF classifier without vs with extra-beat removal."""
    from afscreen.af.experiment import build_table, evaluate
    from afscreen.af.mimic import load_mimic_af

    recs = load_mimic_af(mimic_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, ect in (("no removal", None), ("extra beats removed", rule)):
        df = build_table(recs, 30.0, camera_fs, jitter_ms, ectopy=ect)
        df.to_csv(out / f"mimic_windows_{label.replace(' ', '_')}.csv", index=False)
        res = evaluate(df, thresholds=(0.0, 0.8), models=("forest",))
        res.insert(0, "setting", label)
        rows.append(res)
        if ect is not None:
            per = df.groupby("subject").agg(af=("af", "first"), windows=("window", "size"),
                                            extra_per_window=("n_extra", "mean"),
                                            removed_share=("extra_removed", "mean"))
            per.to_csv(out / "mimic_extra_beats_per_subject.csv")
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "mimic_af_with_removal.csv", index=False)
    return res


# ----------------------------------------------------------------------------- slow / fast heart rate


def hr_flag(hr_bpm: float, slow: float = 50.0, fast: float = 100.0) -> str:
    if not np.isfinite(hr_bpm):
        return "insufficient"
    return "slow" if hr_bpm < slow else ("fast" if hr_bpm > fast else "normal")


def ecg_beats(ecg: np.ndarray, fs: float) -> np.ndarray:
    """R-peak times (s) from the ECG with the XQRS detector of the wfdb package."""
    from wfdb import processing

    from afscreen.signal.filters import fill_gaps

    x = fill_gaps(np.asarray(ecg, dtype=np.float64))
    return processing.xqrs_detect(sig=x, fs=fs, verbose=False) / fs


def run_hr_flags(mimic_root: str | Path, out_dir: str | Path, segment_s: float = 120.0, camera_fs: float = 30.0,
                 jitter_ms: float = 16.0, min_quality: float = 0.8) -> pd.DataFrame:
    """Per 2-minute segment: flag from camera-like PPG beats (confident windows only) vs from ECG."""
    from sklearn.metrics import cohen_kappa_score

    from afscreen.af.experiment import resample
    from afscreen.af.mimic import load_mimic_af
    from afscreen.signal.beats import detect_beats
    from afscreen.signal.filters import clean_pulse, fill_gaps
    from afscreen.signal.quality import beat_shape_consistency

    rows = []
    rng = np.random.default_rng(0)
    for rec in load_mimic_af(mimic_root):
        if rec.ecg is None or np.isnan(rec.ecg).mean() > 0.1:
            continue
        r_peaks = ecg_beats(rec.ecg, rec.fs)
        n = int(segment_s * rec.fs)
        for a in range(0, len(rec.ppg) - n + 1, n):
            seg = rec.ppg[a:a + n]
            if np.isnan(seg).mean() > 0.1:
                continue
            t0 = a / rec.fs
            rp = r_peaks[(r_peaks >= t0) & (r_peaks < t0 + segment_s)]
            ref_hr = 60.0 / np.median(np.diff(rp)) if len(rp) > 10 else np.nan
            x = clean_pulse(resample(fill_gaps(seg), rec.fs, camera_fs), camera_fs)
            hrs = []
            w = int(30 * camera_fs)  # 30 s windows, confident ones only
            for b in range(0, len(x) - w + 1, w):
                beats = detect_beats(x[b:b + w], camera_fs)
                if len(beats) < 6 or beat_shape_consistency(x[b:b + w], camera_fs, beats) < min_quality:
                    continue
                beats = beats + rng.normal(0, jitter_ms / 1000, len(beats))
                hrs.append(60.0 / np.median(np.diff(beats)))
            pred_hr = float(np.median(hrs)) if hrs else np.nan
            rows.append({"subject": rec.id, "af": rec.af, "start_s": t0, "ref_hr": ref_hr, "pred_hr": pred_hr,
                         "ref_flag": hr_flag(ref_hr), "pred_flag": hr_flag(pred_hr)})
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hr_flags_segments.csv", index=False)
    ok = df[(df.ref_flag != "insufficient") & (df.pred_flag != "insufficient")]
    labels = ["slow", "normal", "fast"]
    conf = pd.crosstab(ok.ref_flag, ok.pred_flag).reindex(index=labels, columns=labels, fill_value=0)
    summary = {"segments": len(df), "scored": len(ok), "coverage": len(ok) / max(len(df), 1),
               "accuracy": float((ok.ref_flag == ok.pred_flag).mean()),
               "kappa": float(cohen_kappa_score(ok.ref_flag, ok.pred_flag)),
               "hr_mae_bpm": float(np.nanmean(np.abs(ok.pred_hr - ok.ref_hr)))}
    text = ["# Phase 4: slow / fast heart-rate flags on MIMIC PERform AF\n",
            f"2-minute segments; reference = ECG R-peaks (XQRS); prediction = PPG resampled to {camera_fs:.0f} fps with "
            f"{jitter_ms:.0f} ms beat-timing noise, confident 30 s windows only (beat shape >= {min_quality}). "
            "Thresholds: slow < 50, fast > 100 BPM.\n",
            _md(pd.DataFrame([summary])), "", "Confusion (rows = reference, columns = prediction):\n",
            _md(conf, index=True), ""]
    (out / "hr_flags_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return pd.DataFrame([summary])
