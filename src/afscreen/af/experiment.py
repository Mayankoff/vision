"""Phase 3 - train and validate the AF classifier on MIMIC PERform AF, subject-independently.

    PPG (125 Hz, or resampled to a camera frame rate) -> 30 s windows
        -> beats -> IBI features (features.py)
        -> quality per window: beat-shape consistency (rhythm-independent, used for gating)
                               and spectral peak share (rhythm-dependent, kept for comparison)
    leave-one-subject-out cross-validation of a logistic regression and a random forest,
    for a range of quality thresholds (gating): coverage, sensitivity, specificity, AUC.

The gating sweep is also the plan's Phase 5 check that a regular-rhythm quality measure
(spectral peak share) throws away AF windows, while beat-shape consistency does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from afscreen.af.features import FEATURES, ibi_features
from afscreen.af.mimic import MimicRecording
from afscreen.config import SETTINGS
from afscreen.signal.beats import detect_beats
from afscreen.signal.filters import clean_pulse, fill_gaps
from afscreen.signal.quality import beat_shape_consistency
from afscreen.signal.spectrum import peak_share

MODELS = ("logreg", "forest")


def resample(x: np.ndarray, fs: float, new_fs: float) -> np.ndarray:
    """Polyphase resampling, e.g. 125 Hz -> 30 Hz to mimic a camera's frame rate."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    f = Fraction(new_fs / fs).limit_denominator(1000)
    return resample_poly(x, f.numerator, f.denominator)


def window_rows(rec: MimicRecording, win_s: float = 30.0, camera_fs: float | None = None,
                max_missing: float = 0.1, jitter_ms: float = 0.0, seed: int = 0, ectopy=None) -> list[dict]:
    """One row per non-overlapping ``win_s`` window: features, quality measures, label.

    ``jitter_ms`` adds Gaussian noise of that SD to every beat time before the features are
    computed, to imitate a camera's beat-timing error (training augmentation). With an
    ``ectopy`` rule (flags.ectopy.EctopyRule), extra beats are removed before the features.
    """
    from afscreen.flags.ectopy import ibis_without_extra_beats

    rng = np.random.default_rng(seed)
    x = np.asarray(rec.ppg, dtype=np.float64)
    fs = rec.fs
    rows = []
    n = int(round(win_s * rec.fs))
    for i, a in enumerate(range(0, len(x) - n + 1, n)):
        seg = x[a:a + n]
        if np.isnan(seg).mean() > max_missing or np.nanstd(seg) == 0:
            continue
        seg = fill_gaps(seg)
        f = fs
        if camera_fs:
            seg, f = resample(seg, fs, camera_fs), camera_fs
        p = clean_pulse(seg, f, SETTINGS.hr_band)
        beats = detect_beats(p, f, SETTINGS.hr_band)
        noisy = np.sort(beats + rng.normal(0, jitter_ms / 1000, len(beats))) if jitter_ms else beats
        n_extra, removed = 0, False
        if ectopy is not None:
            ibis, n_extra, removed = ibis_without_extra_beats(noisy, ectopy)
        else:
            ibis = np.diff(noisy)
        feats = ibi_features(ibis)
        if feats is None:
            continue
        rows.append({"subject": rec.id, "af": int(rec.af), "window": i, "start_s": a / fs,
                     "n_extra": n_extra, "extra_removed": removed,
                     "n_beats": len(beats), "hr_bpm": 60.0 / np.mean(np.diff(beats)),
                     "beat_shape": beat_shape_consistency(p, f, beats), "peak_share": peak_share(p, f),
                     **dict(zip(FEATURES, feats))})
    return rows


def build_table(recs: list[MimicRecording], win_s: float = 30.0, camera_fs: float | None = None,
                jitter_ms: float = 0.0, ectopy=None) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(recs):
        rows += window_rows(r, win_s, camera_fs, jitter_ms=jitter_ms, seed=i, ectopy=ectopy)
    return pd.DataFrame(rows)


def make_model(kind: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    if kind == "forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                      random_state=0, n_jobs=-1)
    raise ValueError(kind)


def loso_probabilities(df: pd.DataFrame, kind: str) -> np.ndarray:
    """AF probability of every window, each predicted by a model trained without its subject."""
    prob = np.full(len(df), np.nan)
    x = df[list(FEATURES)].to_numpy()
    y = df["af"].to_numpy()
    g = df["subject"].to_numpy()
    for s in np.unique(g):
        test = g == s
        if len(np.unique(y[~test])) < 2:
            continue
        m = make_model(kind).fit(x[~test], y[~test])
        prob[test] = m.predict_proba(x[test])[:, 1]
    return prob


def _rates(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    pos, neg = y == 1, y == 0
    sens = float(pred[pos].mean()) if pos.any() else float("nan")
    spec = float(1 - pred[neg].mean()) if neg.any() else float("nan")
    return sens, spec


def evaluate(df: pd.DataFrame, gate: str = "beat_shape", thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9),
             models=MODELS) -> pd.DataFrame:
    """Gating sweep: for each quality threshold, train and test (LOSO) on windows that pass it.

    Window-level and subject-level (mean probability over the subject's windows) results, with
    coverage reported separately for AF and non-AF windows.
    """
    from sklearn.metrics import roc_auc_score

    out = []
    n_af, n_non = int((df.af == 1).sum()), int((df.af == 0).sum())
    for thr in thresholds:
        d = df[df[gate] >= thr].reset_index(drop=True)
        row = {"gate": gate, "threshold": thr, "windows": len(d),
               "coverage_af": (d.af == 1).sum() / max(n_af, 1), "coverage_non_af": (d.af == 0).sum() / max(n_non, 1),
               "subjects": d.subject.nunique()}
        for kind in models:
            if d.af.nunique() < 2 or d.subject.nunique() < 3:
                continue
            p = loso_probabilities(d, kind)
            ok = np.isfinite(p)
            y = d.af.to_numpy()[ok]
            sens, spec = _rates(y, p[ok] >= 0.5)
            subj = d[ok].assign(p=p[ok]).groupby("subject").agg(af=("af", "first"), p=("p", "mean"))
            s_sens, s_spec = _rates(subj.af.to_numpy(), subj.p.to_numpy() >= 0.5)
            row |= {f"{kind}_auc": roc_auc_score(y, p[ok]) if len(np.unique(y)) > 1 else float("nan"),
                    f"{kind}_sens": sens, f"{kind}_spec": spec,
                    f"{kind}_subj_sens": s_sens, f"{kind}_subj_spec": s_spec}
        out.append(row)
    return pd.DataFrame(out)


def train_final(df: pd.DataFrame, kind: str = "logreg", min_quality: float = 0.0, gate: str = "beat_shape"):
    """Model fitted on every window that passes the gate (for use on face videos)."""
    d = df[df[gate] >= min_quality]
    return make_model(kind).fit(d[list(FEATURES)].to_numpy(), d["af"].to_numpy())


def run(root: str | Path, out_dir: str | Path, win_s: float = 30.0, camera_fs: float | None = 30.0,
        final_threshold: float = 0.0, jitter_ms: float = 16.0) -> dict[str, pd.DataFrame]:
    """Full Phase 3 experiment on MIMIC PERform AF; writes CSVs, summary.md and the final model.

    Signals: the contact PPG; the PPG resampled to ``camera_fs``; and that with ``jitter_ms``
    of beat-timing noise (the camera's beat error measured on UBFC-rPPG: ~18 ms mean absolute
    IBI error, i.e. ~16 ms per beat). The final model is trained on the last of these.
    """
    import joblib

    from afscreen.af.mimic import load_mimic_af

    recs = load_mimic_af(root)
    if not recs:
        raise SystemExit(f"no MIMIC PERform AF recordings under {root}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    signals = [("contact_125Hz", None, 0.0)]
    if camera_fs:
        signals.append((f"camera_{int(camera_fs)}Hz", camera_fs, 0.0))
        if jitter_ms:
            signals.append((f"camera_{int(camera_fs)}Hz_jitter{int(jitter_ms)}ms", camera_fs, jitter_ms))
    tables, results = {}, {}
    for label, cfs, jit in signals:
        df = build_table(recs, win_s, cfs, jit)
        df.to_csv(out / f"windows_{label}.csv", index=False)
        tables[label] = df
        res = pd.concat([evaluate(df, "beat_shape"), evaluate(df, "peak_share", models=())], ignore_index=True)
        res.insert(0, "signal", label)
        res.to_csv(out / f"gating_{label}.csv", index=False)
        results[label] = res
    final_label = list(tables)[-1]
    model = train_final(tables[final_label], "forest", final_threshold)
    joblib.dump({"model": model, "kind": "forest", "features": FEATURES, "win_s": win_s, "trained_on": final_label,
                 "min_quality": final_threshold, "jitter_ms": signals[-1][2]}, out / "af_classifier.joblib")
    (out / "summary.md").write_text(_summary_md(recs, tables, results), encoding="utf-8")
    return results


def _table_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(lines)


def _summary_md(recs, tables, results) -> str:
    n_af = sum(r.af for r in recs)
    parts = [f"# Phase 3: AF classifier on MIMIC PERform AF\n",
             f"{len(recs)} subjects ({n_af} AF, {len(recs) - n_af} non-AF); 30 s windows; "
             "leave-one-subject-out cross-validation. `subj_*` = per subject (mean window probability).\n"]
    for label, res in results.items():
        df = tables[label]
        parts.append(f"## {label}: {len(df)} windows ({int(df.af.sum())} AF)\n")
        parts.append("Beat-shape gating (rhythm-independent; the proposed confidence part):\n")
        parts.append(_table_md(res[res.gate == "beat_shape"].drop(columns=["signal", "gate"]).dropna(axis=1, how="all")))
        parts.append("\nSpectral peak-share gating (assumes a regular rhythm; for comparison): coverage only\n")
        parts.append(_table_md(res[res.gate == "peak_share"][["threshold", "windows", "coverage_af",
                                                               "coverage_non_af"]]))
        parts.append("")
    return "\n".join(parts) + "\n"
