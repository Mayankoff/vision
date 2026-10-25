"""Step 10 on face videos - AF screening of the fused pulse with the classifier trained on MIMIC.

Each non-overlapping window (same length as in training) gets an AF probability and the
beat-shape quality used in training; windows below the quality threshold are "unreliable"
and get no decision (confidence gating). A video's result is:

    "possible AF"      the mean probability of its reliable windows is >= 0.5
    "regular rhythm"   otherwise
    "unreliable"       no window passed the gate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from afscreen.af.features import ibi_features
from afscreen.signal.beats import detect_beats
from afscreen.signal.quality import beat_shape_consistency

DEFAULT_MODEL = Path("results/phase3/mimic/af_classifier.joblib")


def load_model(path: str | Path = DEFAULT_MODEL) -> dict:
    import joblib

    bundle = joblib.load(path)
    if hasattr(bundle["model"], "n_jobs"):
        bundle["model"].n_jobs = 1  # one window at a time: worker start-up would dominate
    return bundle


def screen_pulse(pulse: np.ndarray, fs: float, bundle: dict, min_quality: float = 0.8, ectopy=None) -> pd.DataFrame:
    """One row per window: AF probability, quality, and whether it passed the gate.

    With an ``ectopy`` rule (flags.ectopy.EctopyRule), extra beats are removed before the features.
    """
    from afscreen.flags.ectopy import ibis_without_extra_beats

    n = int(round(bundle["win_s"] * fs))
    rows = []
    for i, a in enumerate(range(0, len(pulse) - n + 1, n)):
        seg = pulse[a:a + n]
        beats = detect_beats(seg, fs)
        ibis = ibis_without_extra_beats(beats, ectopy)[0] if ectopy is not None else np.diff(beats)
        feats = ibi_features(ibis)
        q = beat_shape_consistency(seg, fs, beats)
        p = float(bundle["model"].predict_proba(feats[None, :])[0, 1]) if feats is not None else float("nan")
        rows.append({"window": i, "start_s": a / fs, "beats": len(beats), "quality": q, "p_af": p,
                     "reliable": bool(feats is not None and q >= min_quality)})
    return pd.DataFrame(rows)


def decide(windows: pd.DataFrame, gated: bool = True) -> str:
    w = windows[windows.reliable] if gated else windows[np.isfinite(windows.p_af)]
    if w.empty:
        return "unreliable"
    return "possible AF" if w.p_af.mean() >= 0.5 else "regular rhythm"


def screen_videos(files: list[Path], bundle: dict, variants=("single", "full"),
                  min_quality: float = 0.8, ectopy=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen cached face-video traces; returns (per-window rows, per-video rows)."""
    from afscreen.pipeline import analyse
    from afscreen.traces import PatchTraces

    wins, vids = [], []
    for f in files:
        tr = PatchTraces.load(f)
        for v in variants:
            w = screen_pulse(analyse(tr, v).pulse, tr.fps, bundle, min_quality, ectopy)
            w.insert(0, "variant", v)
            w.insert(0, "video", tr.meta.get("id", Path(f).stem))
            wins.append(w)
            vids.append({"video": w.video.iloc[0], "variant": v, "windows": len(w), "reliable": int(w.reliable.sum()),
                         "mean_p_af": float(w.p_af.mean()), "gated": decide(w), "ungated": decide(w, False)})
    return pd.concat(wins, ignore_index=True), pd.DataFrame(vids)


def false_alarm_summary(windows: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    """For normal-rhythm videos: share of windows and videos called 'possible AF', and coverage."""
    rows = []
    for v, w in windows.groupby("variant", sort=False):
        ok = w[np.isfinite(w.p_af)]
        rel = w[w.reliable]
        vv = videos[videos.variant == v]
        rows.append({"variant": v, "videos": len(vv), "windows": len(w),
                     "window_FA_ungated": float((ok.p_af >= 0.5).mean()) if len(ok) else float("nan"),
                     "window_FA_gated": float((rel.p_af >= 0.5).mean()) if len(rel) else float("nan"),
                     "coverage": len(rel) / max(len(w), 1),
                     "video_FA_ungated": float((vv.ungated == "possible AF").mean()),
                     "video_FA_gated": float((vv.gated == "possible AF").mean())})
    return pd.DataFrame(rows)
