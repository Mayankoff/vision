"""Step 5 - the four per-patch trust checks, computed in one window.

All four avoid assuming a regular rhythm: they ask "does this patch see the face well?" and
"does it see the *same* pulse as the other patches?", never "is its spectrum one sharp peak?".
"""

from __future__ import annotations

import numpy as np

from afscreen.config import SETTINGS
from afscreen.signal.beats import beat_agreement, detect_beats


def facing_weight(facing: np.ndarray, min_facing: float = SETTINGS.min_facing) -> np.ndarray:
    """Weight = |cos| of the angle to the camera (Lambert's law); 0 when turned too far away."""
    f = np.nan_to_num(np.asarray(facing, dtype=np.float64), nan=0.0)
    return np.where(f < min_facing, 0.0, f)


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def brightness_weight(luma: np.ndarray, dark=SETTINGS.dark_luma, bright=SETTINGS.bright_luma) -> np.ndarray:
    """0 in deep shadow or near saturation (shine clips the pulse), 1 in between, with smooth ramps."""
    y = np.nan_to_num(np.asarray(luma, dtype=np.float64), nan=0.0)
    return _ramp(y, *dark) * (1.0 - _ramp(y, *bright))


def luma(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma of RGB values (..., 3)."""
    return np.asarray(rgb, dtype=np.float64) @ np.array([0.299, 0.587, 0.114])


def consistency(seg: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Correlation (0-1) of each patch pulse with the weighted pulse of all *other* patches.

    ``seg`` (win, P) holds unit-variance patch pulses; ``w`` (P,) the weights used to form the
    leave-one-out consensus. In AF every patch shows the same irregular beats, so consistency
    stays high; noise differs from patch to patch, so it drops.
    """
    seg = np.asarray(seg, dtype=np.float64)
    n_p = seg.shape[1]
    if n_p < 2:
        return np.ones(n_p)
    total = seg @ w
    others = total[:, None] - seg * w[None, :]  # (win, P): consensus without patch k
    a = seg - seg.mean(axis=0)
    b = others - others.mean(axis=0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, (a * b).sum(axis=0) / den, 0.0)
    return np.clip(r, 0.0, 1.0)


def beat_agreement_scores(seg: np.ndarray, w: np.ndarray, fs: float, tol_s: float = SETTINGS.beat_tol_s) -> np.ndarray:
    """Share (0-1) of each patch's beats that line up with the beats of the leave-one-out consensus."""
    seg = np.asarray(seg, dtype=np.float64)
    n_p = seg.shape[1]
    if n_p < 2:
        return np.ones(n_p)
    total = seg @ w
    out = np.zeros(n_p)
    for k in range(n_p):
        if w[k] <= 0 and not np.any(seg[:, k]):
            continue
        consensus = total - w[k] * seg[:, k]
        if not np.any(consensus):
            continue
        out[k] = beat_agreement(detect_beats(seg[:, k], fs), detect_beats(consensus, fs), tol_s)
    return out
