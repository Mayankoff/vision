"""Beat detection and beat matching."""

from __future__ import annotations

import numpy as np
from scipy import signal

from afscreen.config import SETTINGS
from afscreen.signal.spectrum import hr_fft


def refine_peaks(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Sub-sample peak positions from a parabola through each peak and its neighbours.

    Without this, beat intervals are quantised to whole frames (33 ms at 30 fps), which alone
    adds irregularity that looks like AF.
    """
    peaks = np.asarray(peaks, dtype=int)
    p = peaks[(peaks > 0) & (peaks < len(x) - 1)]
    a, b, c = x[p - 1], x[p], x[p + 1]
    denom = a - 2 * b + c
    offset = np.where(denom != 0, 0.5 * (a - c) / np.where(denom != 0, denom, 1), 0.0)
    return p + np.clip(offset, -0.5, 0.5)


def refine_peaks_spline(x: np.ndarray, peaks: np.ndarray, factor: int = 10) -> np.ndarray:
    """Sub-sample peak positions as the maximum of a cubic spline through +/- 2 samples."""
    from scipy.interpolate import CubicSpline

    peaks = np.asarray(peaks, dtype=int)
    p = peaks[(peaks > 1) & (peaks < len(x) - 2)]
    out = np.empty(len(p))
    fine = np.linspace(-1, 1, 2 * factor + 1)
    for j, k in enumerate(p):
        cs = CubicSpline(np.arange(-2, 3), x[k - 2:k + 3])
        out[j] = k + fine[np.argmax(cs(fine))]
    return out


def detect_beats(x: np.ndarray, fs: float, band=SETTINGS.hr_band, min_period_share: float = 0.5,
                 refine: str = "parabolic") -> np.ndarray:
    """Beat times in seconds (systolic peaks) of a band-passed pulse signal.

    Peaks closer than ``min_period_share`` of the dominant period are suppressed so the
    secondary bump of each beat is not counted twice. The share is lower than for a regular
    rhythm (0.5 rather than ~0.6) so that the short intervals of an irregular rhythm survive.
    ``refine``: "parabolic" (default), "spline", or "none" (whole samples).
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0:
        return np.zeros(0)
    period = 60.0 / hr_fft(x, fs, band) * fs
    distance = max(1, int(max(fs / band[1], min_period_share * period)))
    peaks, _ = signal.find_peaks(x, distance=distance, prominence=0.25 * np.std(x))
    if refine == "none":
        return peaks.astype(np.float64) / fs
    if refine == "spline":
        return refine_peaks_spline(x, peaks) / fs
    return refine_peaks(x, peaks) / fs


def intervals(beat_times: np.ndarray) -> np.ndarray:
    """Inter-beat intervals (s)."""
    return np.diff(np.asarray(beat_times, dtype=np.float64))


def match_count(a: np.ndarray, b: np.ndarray, tol_s: float) -> int:
    """Number of one-to-one pairs between two sorted beat-time lists that are within ``tol_s``."""
    a, b = np.asarray(a), np.asarray(b)
    i = j = n = 0
    while i < len(a) and j < len(b):
        d = a[i] - b[j]
        if abs(d) <= tol_s:
            n, i, j = n + 1, i + 1, j + 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return n


def beat_agreement(a: np.ndarray, b: np.ndarray, tol_s: float = SETTINGS.beat_tol_s) -> float:
    """F1-style agreement (0-1) between two beat-time lists: 2 x matched / (len(a) + len(b))."""
    total = len(a) + len(b)
    return 2.0 * match_count(a, b, tol_s) / total if total else 0.0
