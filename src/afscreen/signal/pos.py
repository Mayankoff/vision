"""POS - Plane-Orthogonal-to-Skin (Wang, den Brinker, Stuijk & de Haan, 2017)."""

from __future__ import annotations

import math

import numpy as np

from afscreen.config import SETTINGS

_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def pos(rgb: np.ndarray, fs: float, win_s: float = SETTINGS.pos_window_s) -> np.ndarray:
    """Pulse signal (T,) from an RGB trace (T, 3).

    In each short window the temporally normalised RGB is projected onto two axes orthogonal
    to the (1, 1, 1) intensity direction; the two projections are combined with an
    alpha-tuning step that cancels the remaining distortion. Windows are overlap-added.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    n = rgb.shape[0]
    win = min(n, math.ceil(win_s * fs))
    h = np.zeros(n)
    for end in range(win, n + 1):
        start = end - win
        seg = rgb[start:end]
        mean = seg.mean(axis=0)
        if np.any(mean <= 0):
            continue
        s = (seg / mean) @ _PROJECTION.T
        s1, s2 = s[:, 0], s[:, 1]
        std2 = s2.std()
        p = s1 + (s1.std() / std2 if std2 > 0 else 0.0) * s2
        h[start:end] += p - p.mean()
    return h
