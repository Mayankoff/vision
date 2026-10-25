"""Phase 2 experiments - breathing rate, breathing-aware heart rate, beat timing and HRV.

    breathing   MIMIC PERform Testing adults (100) + MIMIC PERform AF recordings with respiration:
                camera-like PPG (resampled to 30 fps) vs impedance pneumography, per window,
                for two pulse bands, three window lengths and the fusion threshold
    heart rate  same recordings, 30 s windows, vs ECG R-peaks: plain FFT, breathing-aware FFT,
                beats (60 / mean IBI), and the combination (beats when the rhythm is irregular)
    HRV         UBFC face videos with a reference: IBI, RMSSD and SDNN errors for three ways
                of placing beats between frames (none, parabolic, cubic spline)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from afscreen.af.features import nrmssd as _nrmssd
from afscreen.config import SETTINGS
from afscreen.signal.beats import detect_beats
from afscreen.signal.filters import bandpass, clean_pulse, fill_gaps
from afscreen.signal.resp import modulation_series, reference_rate, series_rate
from afscreen.signal.spectrum import hr_fft, hr_fft_breathing_aware

CAMERA_FS = 30.0
WIDE_BAND = (0.1, 3.3)
IRREGULAR_NRMSSD = 0.1  # above this the rhythm counts as irregular (RIFV dropped, HR from beats)
FUSION_SDS = (1.0, 2.0, 3.0, 4.0, np.inf)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(lines)


def _irregular(beats: np.ndarray) -> bool:
    ibis = np.diff(beats)
    return len(ibis) >= 4 and _nrmssd(ibis) > IRREGULAR_NRMSSD


def _recordings(mimic_root):
    from afscreen.af.mimic import load_mimic_af, load_mimic_testing

    for r in load_mimic_testing(mimic_root, "a"):
        yield "ICU adults (Testing)", r
    for r in load_mimic_af(mimic_root):
        yield ("ICU AF" if r.af else "ICU non-AF"), r


# ----------------------------------------------------------------------------- breathing


def breathing_rows(rec, group: str, win_s: float) -> list[dict]:
    """Per window: reference rate and every modulation's estimate from both pulse bands."""
    if rec.resp is None:
        return []
    rows = []
    n = int(win_s * rec.fs)
    for i, a in enumerate(range(0, len(rec.ppg) - n + 1, n)):
        seg = rec.ppg[a:a + n]
        if np.isnan(seg).mean() > 0.1:
            continue
        ref = reference_rate(rec.resp[a:a + n], rec.fs)
        if not np.isfinite(ref):
            continue
        from afscreen.af.experiment import resample

        raw = resample(fill_gaps(seg), rec.fs, CAMERA_FS)
        x = clean_pulse(raw, CAMERA_FS)
        beats = detect_beats(x, CAMERA_FS)
        if len(beats) < 6:
            continue
        wide = bandpass(raw - raw.mean(), CAMERA_FS, WIDE_BAND)
        row = {"group": group, "record": rec.id, "window_s": win_s, "start_s": a / rec.fs, "ref": ref,
               "irregular": _irregular(beats)}
        for band, sig in (("hr", x), ("wide", wide)):
            for k, (t, v) in modulation_series(sig, CAMERA_FS, beats, with_intensity=True).items():
                row[f"{band}_{k}"] = series_rate(t, v, win_s)
        rows.append(row)
    return rows


def fuse_estimates(df: pd.DataFrame, band: str, mods: tuple[str, ...], max_sd: float) -> pd.Series:
    """Smart fusion of the chosen modulations per row (RIFV dropped in irregular windows)."""
    cols = [f"{band}_{m}" for m in mods]
    vals = df[cols].to_numpy(dtype=np.float64).copy()
    if "RIFV" in mods:
        vals[df.irregular.to_numpy(), mods.index("RIFV")] = np.nan
    n = np.isfinite(vals).sum(axis=1)
    mean = np.where(n > 0, np.nanmean(np.where(np.isfinite(vals), vals, np.nan), axis=1), np.nan)
    sd = np.where(n >= 2, np.nanstd(vals, axis=1), 0.0)
    return pd.Series(np.where((n > 0) & (sd <= max_sd), mean, np.nan), index=df.index)


METHODS = {  # name: (band, modulations)
    "camera POS pulse: RIAV + RIFV": ("hr", ("RIAV", "RIFV")),
    "wide-band: RIAV + RIFV": ("wide", ("RIAV", "RIFV")),
    "wide-band: RIAV + RIFV + RIIV": ("wide", ("RIAV", "RIFV", "RIIV")),
    "wide-band: RIAV only": ("wide", ("RIAV",)),
}


def breathing_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (w, g), d in df.groupby(["window_s", "group"]):
        for name, (band, mods) in METHODS.items():
            for sd in FUSION_SDS:
                e = fuse_estimates(d, band, mods, sd) - d.ref
                ok = np.isfinite(e)
                out.append({"window_s": w, "group": g, "method": name, "max_sd": sd, "windows": len(d),
                            "coverage": float(ok.mean()), "MAE": float(np.abs(e[ok]).mean()) if ok.any() else np.nan,
                            "within_2": float((np.abs(e[ok]) <= 2).mean()) if ok.any() else np.nan})
    return pd.DataFrame(out)


def run_breathing(mimic_root, out_dir, windows=(32.0, 64.0)) -> pd.DataFrame:
    rows = []
    for group, rec in _recordings(mimic_root):
        for w in windows:
            rows += breathing_rows(rec, group, w)
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "breathing_windows.csv", index=False)
    summ = breathing_summary(df)
    summ.to_csv(out / "breathing_summary.csv", index=False)
    return summ


# ----------------------------------------------------------------------------- heart rate


def heart_rate_rows(rec, group: str, win_s: float = 30.0) -> list[dict]:
    from afscreen.af.experiment import resample
    from afscreen.flags.experiment import ecg_beats
    from afscreen.signal.resp import breathing_rate

    if rec.ecg is None or np.isnan(rec.ecg).mean() > 0.1:
        return []
    r_peaks = ecg_beats(rec.ecg, rec.fs)
    rows = []
    n = int(win_s * rec.fs)
    for a in range(0, len(rec.ppg) - n + 1, n):
        seg = rec.ppg[a:a + n]
        t0 = a / rec.fs
        rp = r_peaks[(r_peaks >= t0) & (r_peaks < t0 + win_s)]
        if np.isnan(seg).mean() > 0.1 or len(rp) < 6:
            continue
        rr = np.diff(rp)
        rr = rr[(rr > 0.25) & (rr < 2.5)]
        if len(rr) < 5:
            continue
        ref = 60.0 / rr.mean()
        raw = resample(fill_gaps(seg), rec.fs, CAMERA_FS)
        x = clean_pulse(raw, CAMERA_FS)
        beats = detect_beats(x, CAMERA_FS)
        if len(beats) < 4:
            continue
        irregular = _irregular(beats)
        wide = bandpass(raw - raw.mean(), CAMERA_FS, WIDE_BAND)
        br, _ = breathing_rate(wide, CAMERA_FS, beats, irregular=irregular, with_intensity=True)
        fft = hr_fft(x, CAMERA_FS, SETTINGS.hr_band)
        aware = hr_fft_breathing_aware(x, CAMERA_FS, br, SETTINGS.hr_band)
        beat_hr = 60.0 / np.mean(np.diff(beats))
        rows.append({"group": group, "record": rec.id, "start_s": t0, "ref": ref, "irregular": irregular,
                     "breathing": br, "fft": fft, "breathing_aware": aware, "beats": beat_hr,
                     "combined": beat_hr if irregular else aware})
    return rows


def run_heart_rate(mimic_root, out_dir) -> pd.DataFrame:
    rows = []
    for group, rec in _recordings(mimic_root):
        rows += heart_rate_rows(rec, group)
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hr_windows.csv", index=False)
    summ = []
    for g, d in [*df.groupby("group"), ("all", df)]:
        row = {"group": g, "windows": len(d), "irregular_share": float(d.irregular.mean())}
        for m in ("fft", "breathing_aware", "beats", "combined"):
            e = np.abs(d[m] - d.ref)
            row[f"{m}_MAE"] = float(e.mean())
            row[f"{m}_over10"] = float((e > 10).mean())
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "hr_summary.csv", index=False)
    return summ


# ----------------------------------------------------------------------------- HRV on face videos


def run_hrv(cache_dirs, out_dir, variants=("single", "full")) -> pd.DataFrame:
    from afscreen.eval.metrics import beat_f1, ibi_error_ms
    from afscreen.eval.reference import align_reference, reference_pulse
    from afscreen.pipeline import analyse
    from afscreen.signal.hrv import rmssd_ms, sdnn_ms
    from afscreen.traces import PatchTraces

    files = sorted({f for d in cache_dirs for f in Path(d).glob("*.npz")})
    rows = []
    for f in files:
        tr = PatchTraces.load(f)
        if tr.ref_pulse is None:
            continue
        ref = reference_pulse(tr.ref_pulse, tr.fps)
        for v in variants:
            r = analyse(tr, v)
            aligned, _, _ = align_reference(ref, r.pulse, tr.fps)
            ref_beats = detect_beats(aligned, tr.fps, refine="spline")
            ref_ibis = np.diff(ref_beats)
            for refine in ("none", "parabolic", "spline"):
                b = detect_beats(r.pulse, tr.fps, refine=refine)
                ibis = np.diff(b)
                rows.append({"video": tr.meta.get("id", f.stem), "variant": v, "refine": refine, "fps": tr.fps,
                             "ibi_err_ms": ibi_error_ms(b, ref_beats), "beat_f1": beat_f1(b, ref_beats),
                             "rmssd_ref": rmssd_ms(ref_ibis), "rmssd": rmssd_ms(ibis),
                             "sdnn_ref": sdnn_ms(ref_ibis), "sdnn": sdnn_ms(ibis)})
    df = pd.DataFrame(rows)
    df["rmssd_err"] = (df.rmssd - df.rmssd_ref).abs()
    df["sdnn_err"] = (df.sdnn - df.sdnn_ref).abs()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hrv_per_video.csv", index=False)
    summ = (df.groupby(["variant", "refine"], sort=False)
            .agg(videos=("video", "size"), ibi_err_ms=("ibi_err_ms", "mean"), beat_f1=("beat_f1", "mean"),
                 rmssd_err_ms=("rmssd_err", "mean"), sdnn_err_ms=("sdnn_err", "mean"),
                 rmssd_ref_ms=("rmssd_ref", "mean"), rmssd_ms=("rmssd", "mean"))
            .reset_index())
    summ.to_csv(out / "hrv_summary.csv", index=False)
    return summ


def write_summary(out_dir, breathing: pd.DataFrame | None, hr: pd.DataFrame | None, hrv: pd.DataFrame | None) -> None:
    parts = ["# Phase 2: breathing, heart rate, beat timing and HRV\n"]
    if breathing is not None:
        best = breathing[breathing.max_sd == 4.0]
        parts += ["## Breathing rate (camera-like PPG vs impedance respiration; smart fusion SD <= 4 /min)\n",
                  _md(best.drop(columns=["max_sd"])), "",
                  "Fusion threshold sweep (64 s, wide-band, three modulations):\n",
                  _md(breathing[(breathing.window_s == 64) & (breathing.method == "wide-band: RIAV + RIFV + RIIV")]
                      .drop(columns=["window_s", "method"])), ""]
    if hr is not None:
        parts += ["## Heart rate per 30 s window vs ECG (camera-like PPG)\n", _md(hr), ""]
    if hrv is not None:
        parts += ["## Beat timing and HRV on UBFC face videos (about 1 min each)\n", _md(hrv), ""]
    # SUMMARY.md holds the hand-written interpretation; the generated tables go next to it.
    (Path(out_dir) / "summary_tables.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
