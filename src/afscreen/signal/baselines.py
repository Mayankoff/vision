"""Comparison baselines for the pulse-extraction step (Phase 5). Not part of the proposed method.

    GREEN   the normalised green channel (Verkruysse et al., 2008), inverted so that more blood
            gives a higher value
    CHROM   chrominance-based (de Haan & Jeanne, 2013): two colour-difference signals combined
            with a per-window ratio of their standard deviations

Both take an RGB trace (T, 3) and return a pulse signal (T,), like :func:`afscreen.signal.pos.pos`.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from afscreen.config import SETTINGS


def green(rgb: np.ndarray, fs: float, win_s: float = SETTINGS.pos_window_s) -> np.ndarray:
    g = np.asarray(rgb, dtype=np.float64)[:, 1]
    return -(g / g.mean() - 1.0)


def chrom(rgb: np.ndarray, fs: float, win_s: float = SETTINGS.pos_window_s, band=SETTINGS.hr_band) -> np.ndarray:
    """CHROM with Hann-windowed overlap-add of half-overlapping windows (as in the original paper)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    n = rgb.shape[0]
    win = min(n, max(8, math.ceil(win_s * fs)))
    win += win % 2
    step = win // 2
    sos = signal.butter(SETTINGS.filter_order, band, btype="bandpass", fs=fs, output="sos")
    hann = np.hanning(win)
    h = np.zeros(n)
    starts = list(range(0, max(1, n - win + 1), step))
    for a in starts:
        seg = rgb[a:a + win]
        mean = seg.mean(axis=0)
        if np.any(mean <= 0) or len(seg) < win:
            continue
        c = seg / mean
        xs = 3 * c[:, 0] - 2 * c[:, 1]
        ys = 1.5 * c[:, 0] + c[:, 1] - 1.5 * c[:, 2]
        pad = min(len(xs) - 1, 3 * (2 * len(sos) + 1))
        xf = signal.sosfiltfilt(sos, xs, padlen=pad)
        yf = signal.sosfiltfilt(sos, ys, padlen=pad)
        alpha = xf.std() / yf.std() if yf.std() > 0 else 0.0
        s = xf - alpha * yf
        h[a:a + win] += hann * (s - s.mean())
    return h


PULSE_METHODS = {"green": green, "chrom": chrom}
