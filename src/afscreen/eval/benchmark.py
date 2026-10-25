"""Score every variant on a folder of cached traces (Phase 1 exit check and the Phase 5 ablation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from afscreen.eval.metrics import beat_f1, ibi_error_ms, snr_db, summarize
from afscreen.eval.reference import align_reference, reference_beats, reference_pulse
from afscreen.pipeline import VARIANTS, analyse, window_hr
from afscreen.signal.beats import detect_beats
from afscreen.signal.spectrum import hr_fft
from afscreen.traces import PatchTraces


def score_video(tr: PatchTraces, variants=tuple(VARIANTS), level: int | None = None) -> list[dict]:
    """One row per variant: whole-video HR error, windowed HR error, IBI error, SNR, confidence."""
    if tr.ref_pulse is None:
        raise ValueError(f"{tr.meta.get('id', '?')}: no reference pulse in the cached traces")
    fs = tr.fps
    ref = reference_pulse(tr.ref_pulse, fs)
    ref_hr = hr_fft(ref, fs)
    ref_beats = tr.ref_beats if tr.ref_beats is not None else reference_beats(tr.ref_pulse, fs)
    rows = []
    base = {"dataset": tr.meta.get("dataset", ""), "video": tr.meta.get("id", ""), "hr_ref": ref_hr,
            "detection_rate": tr.detection_rate}
    if tr.detection_rate < 0.5:  # face mostly not found: report the failure instead of a number
        return [{**base, "variant": v, "level": 0} for v in variants]
    for v in variants:
        r = analyse(tr, v, level)
        ref_win = window_hr(ref, fs, r.window_t)
        lag_s = 0.0
        if tr.ref_beats is None:  # beats from a contact sensor: align it to the face pulse first
            aligned, lag_s, _ = align_reference(ref, r.pulse, fs)
            ref_beats = detect_beats(aligned, fs)
        rows.append({
            **base, "variant": v, "level": r.level, "hr_pred": r.hr_bpm, "abs_err": abs(r.hr_bpm - ref_hr),
            "hr_beats": r.hr_beats_bpm,
            "win_mae": float(np.mean(np.abs(r.window_hr - ref_win))),
            "ibi_err_ms": ibi_error_ms(r.beats, ref_beats),
            "beat_f1": beat_f1(r.beats, ref_beats),
            "snr_db": snr_db(r.pulse, fs, ref_hr),
            "confidence": r.mean_confidence,
            "ref_lag_s": lag_s,
        })
    return rows


def run(cache_dir: str | Path, out_dir: str | Path, variants=tuple(VARIANTS), level: int | None = None) -> pd.DataFrame:
    """Score all ``*.npz`` traces in ``cache_dir``; writes per_video.csv, summary.csv and summary.md."""
    files = sorted(Path(cache_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"no cached traces in {cache_dir} (run 'afscreen extract' first)")
    rows = []
    for f in files:
        rows += score_video(PatchTraces.load(f), variants, level)
    per_video = pd.DataFrame(rows)
    summary = []
    for v in variants:
        d = per_video[per_video.variant == v]
        m = summarize(d.hr_pred, d.hr_ref)
        summary.append({"variant": v, "videos": m["n"], "MAE": m["MAE"], "MAE_se": m["MAE_se"], "RMSE": m["RMSE"],
                        "MAPE": m["MAPE"], "Pearson": m["Pearson"], "win_MAE": d.win_mae.mean(),
                        "IBI_err_ms": d.ibi_err_ms.median(), "beat_F1": d.beat_f1.mean(), "SNR_dB": d.snr_db.mean(),
                        "confidence": d.confidence.mean()})
    summary = pd.DataFrame(summary)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(out / "per_video.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(lines) + "\n"
