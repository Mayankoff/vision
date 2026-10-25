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
    from numpy.lib.stride_tricks import sliding_window_view

    rgb = np.asarray(rgb, dtype=np.float64)
    n = rgb.shape[0]
    win = min(n, math.ceil(win_s * fs))
    # All windows at once (vectorised form of the per-window loop; identical result).
    seg = sliding_window_view(rgb, win, axis=0).transpose(0, 2, 1)  # (n - win + 1, win, 3)
    mean = seg.mean(axis=1, keepdims=True)
    ok = np.all(mean[:, 0, :] > 0, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (seg / mean) @ _PROJECTION.T  # (W, win, 2)
    s1, s2 = s[..., 0], s[..., 1]
    std1, std2 = s1.std(axis=1), s2.std(axis=1)
    alpha = np.where(std2 > 0, std1 / np.where(std2 > 0, std2, 1.0), 0.0)
    p = s1 + alpha[:, None] * s2
    p = np.where(ok[:, None], p - p.mean(axis=1, keepdims=True), 0.0)
    h = np.zeros(n)
    for k in range(win):  # overlap-add: window w covers samples w .. w + win - 1
        h[k:k + len(p)] += p[:, k]
    return h
