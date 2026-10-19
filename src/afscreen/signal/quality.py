"""Rhythm-independent pulse quality measures.

None of these assume a regular heartbeat, so an irregular rhythm (AF) is not mistaken for noise.
"""

from __future__ import annotations

import numpy as np

from afscreen.signal.beats import detect_beats


def beat_shape_consistency(x: np.ndarray, fs: float, beats: np.ndarray | None = None, n_points: int = 32) -> float:
    """Mean correlation (0-1) of each beat's waveform with the average beat (template matching).

    A standard PPG signal-quality index (Orphanidou et al., 2015). AF changes the *timing* of
    beats, not their basic shape, so a clean AF pulse still scores high; noise does not.
    Each beat is taken from 35% of the median interval before its peak to 65% after it.
    """
    x = np.asarray(x, dtype=np.float64)
    beats = detect_beats(x, fs) if beats is None else np.asarray(beats)
    if len(beats) < 3:
        return 0.0
    m = float(np.median(np.diff(beats)))
    grid = np.linspace(-0.35 * m, 0.65 * m, n_points)
    t = np.arange(len(x)) / fs
    segs = [np.interp(b + grid, t, x) for b in beats if b + grid[0] >= 0 and b + grid[-1] <= t[-1]]
    if len(segs) < 3:
        return 0.0
    segs = np.array(segs)
    segs = segs - segs.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(segs, axis=1)
    segs = segs[norms > 0] / norms[norms > 0, None]
    if len(segs) < 3:
        return 0.0
    template = segs.mean(axis=0)
    tn = np.linalg.norm(template)
    if tn == 0:
        return 0.0
    return float(np.clip(np.mean(segs @ (template / tn)), 0.0, 1.0))
