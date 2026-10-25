"""Step 8 - breathing rate from the pulse signal (Karlen et al., 2013).

Breathing modulates the pulse in three ways; each gives one beat-by-beat series:

    RIIV  intensity: the level of each beat's foot (baseline wander; needs a pulse that keeps
          frequencies below the heart-rate band, so it is not available from the band-passed
          camera pulse)
    RIAV  amplitude: peak minus foot of each beat
    RIFV  frequency: the inter-beat intervals (respiratory sinus arrhythmia)

Each series is resampled to 4 Hz, band-passed to the breathing band and its spectral peak taken.
Smart fusion: if the estimates agree (SD <= 4 breaths/min) their mean is the rate, otherwise
the window has no breathing rate. In AF the beat intervals are random, so RIFV carries no
breathing and is left out when the rhythm is irregular.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

RESP_BAND = (0.1, 0.6)  # Hz: 6-36 breaths/min
SERIES_FS = 4.0
FUSION_MAX_SD = 4.0  # breaths/min (Karlen et al., 2013)


def beat_feet(x: np.ndarray, fs: float, beats: np.ndarray) -> np.ndarray:
    """Sample index of each beat's foot: the minimum between the previous beat and this one."""
    peaks = np.round(np.asarray(beats) * fs).astype(int)
    feet = np.empty(len(peaks), dtype=int)
    for i, p in enumerate(peaks):
        lo = peaks[i - 1] if i > 0 else max(0, p - int(fs))
        seg = x[lo:p + 1]
        feet[i] = lo + int(np.argmin(seg)) if len(seg) else p
    return feet


def modulation_series(x: np.ndarray, fs: float, beats: np.ndarray, with_intensity: bool = False) -> dict:
    """{name: (times, values)} for RIAV, RIFV (and RIIV if ``with_intensity``)."""
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 4:
        return {}
    peaks = np.clip(np.round(beats * fs).astype(int), 0, len(x) - 1)
    feet = beat_feet(x, fs, beats)
    out = {"RIAV": (beats, x[peaks] - x[feet]),
           "RIFV": (beats[1:], np.diff(beats))}
    if with_intensity:
        out["RIIV"] = (beats, x[feet])
    return out


def series_rate(t: np.ndarray, v: np.ndarray, duration: float, band=RESP_BAND) -> float:
    """Breaths/min from one beat-by-beat series (spectral peak in ``band``), or NaN."""
    if len(t) < 6 or np.std(v) == 0:
        return float("nan")
    grid = np.arange(t[0], t[-1], 1 / SERIES_FS)
    if len(grid) < 16 or (t[-1] - t[0]) < 0.6 * duration:
        return float("nan")
    y = np.interp(grid, t, v)
    y = signal.detrend(y)
    sos = signal.butter(2, band, btype="bandpass", fs=SERIES_FS, output="sos")
    y = signal.sosfiltfilt(sos, y, padlen=min(len(y) - 1, 15))
    f, p = signal.periodogram(y, fs=SERIES_FS, nfft=max(1024, 1 << int(np.ceil(np.log2(len(y))))), detrend=False)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or p[m].max() <= 0:
        return float("nan")
    return float(f[m][np.argmax(p[m])] * 60.0)


def breathing_rate(x: np.ndarray, fs: float, beats: np.ndarray, irregular: bool = False,
                   with_intensity: bool = False, max_sd: float = FUSION_MAX_SD) -> tuple[float, dict]:
    """(fused breaths/min or NaN, per-modulation estimates) for one window of pulse ``x``."""
    duration = len(x) / fs
    est = {k: series_rate(t, v, duration) for k, (t, v) in modulation_series(x, fs, beats, with_intensity).items()}
    if irregular:
        est.pop("RIFV", None)
    vals = np.array([v for v in est.values() if np.isfinite(v)])
    if len(vals) == 0:
        return float("nan"), est
    if len(vals) >= 2 and np.std(vals) > max_sd:
        return float("nan"), est
    return float(np.mean(vals)), est


def reference_rate(resp: np.ndarray, fs: float, band=RESP_BAND, agree_bpm: float = 2.0) -> float:
    """Breaths/min from a respiration signal: spectral peak, kept only if a breath count
    (rising zero crossings of the band-passed signal) agrees within ``agree_bpm``; else NaN."""
    r = np.asarray(resp, dtype=np.float64)
    if np.isnan(r).mean() > 0.1 or np.nanstd(r) == 0:
        return float("nan")
    from afscreen.signal.filters import fill_gaps

    r = fill_gaps(r)
    sos = signal.butter(2, band, btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, r - r.mean())
    f, p = signal.periodogram(y, fs=fs, nfft=max(1 << 14, 1 << int(np.ceil(np.log2(len(y))))), detrend=False)
    m = (f >= band[0]) & (f <= band[1])
    spec = f[m][np.argmax(p[m])] * 60.0
    up = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0))
    if len(up) < 3:
        return float("nan")
    count = 60.0 * (len(up) - 1) / ((up[-1] - up[0]) / fs)
    return float(spec) if abs(spec - count) <= agree_bpm else float("nan")
