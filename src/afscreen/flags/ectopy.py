"""Step 11 - extra (premature) beats from beat timing alone.

An extra beat comes early and is followed by a pause, inside an otherwise regular rhythm:

    beat i is premature  if  IBI_before < (1 - prematurity) x local median IBI
                         and IBI_after  > recovery x local median IBI

The local median is taken over the surrounding intervals, leaving out the two around beat i.
Removing extra beats before the AF features stops isolated ectopy from looking like AF; when
too many beats are flagged in a window the rhythm is not "otherwise regular", so nothing is
removed and the AF classifier sees the irregularity as it is.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EctopyRule:
    prematurity: float = 0.15  # IBI before must be this much shorter than the local median
    recovery: float = 0.95  # IBI after must be at least this share of the local median
    context: int = 8  # intervals on each side for the local median
    max_share: float = 0.2  # if more beats than this are flagged, the rhythm is not "otherwise regular"
    # Removal only: an extra beat is removed only when the intervals just outside its short-long
    # pair are within this share of the local median ("inside an otherwise regular rhythm").
    # In AF the neighbours are irregular too, so AF beats are left alone. None = no check.
    neighbour_tol: float | None = None


def local_median(ibis: np.ndarray, context: int) -> np.ndarray:
    """Median of up to ``context`` intervals on each side of every beat ``i`` (beat i sits
    between ``ibis[i-1]`` and ``ibis[i]``), leaving out those two intervals. NaN where fewer
    than 3 intervals are available; index 0 and ``len(ibis)`` (first and last beat) are NaN."""
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(ibis)
    out = np.full(n + 1, np.nan)
    if n < 2:
        return out
    pad = np.full(context, np.nan)
    win = sliding_window_view(np.concatenate([pad, ibis, pad]), 2 * context + 2)[: n - 1]
    both = np.concatenate([win[:, :context], win[:, context + 2:]], axis=1)  # drop ibis[i-1], ibis[i]
    enough = np.isfinite(both).sum(axis=1) >= 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows; masked out below
        med = np.nanmedian(both, axis=1)
    out[1:n] = np.where(enough, med, np.nan)
    return out


def detect_extra_beats(beats: np.ndarray, rule: EctopyRule = EctopyRule()) -> np.ndarray:
    """Indices (into ``beats``) of premature beats."""
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 5:
        return np.zeros(0, dtype=int)
    ibis = np.diff(beats)
    ref = local_median(ibis, rule.context)
    idx = np.arange(1, len(beats) - 1)
    before, after, m = ibis[idx - 1], ibis[idx], ref[idx]
    with np.errstate(invalid="ignore"):
        hit = (before < (1 - rule.prematurity) * m) & (after > rule.recovery * m)
    return idx[hit]


def ibis_without_extra_beats(beats: np.ndarray, rule: EctopyRule = EctopyRule()) -> tuple[np.ndarray, int, bool]:
    """IBIs with the two intervals around each extra beat dropped.

    Returns (IBIs, number of extra beats, whether they were removed). Nothing is removed when
    more than ``rule.max_share`` of the beats are flagged (the rhythm is not otherwise regular).
    """
    beats = np.asarray(beats, dtype=np.float64)
    ibis = np.diff(beats)
    extra = detect_extra_beats(beats, rule)
    if len(extra) == 0 or len(extra) > rule.max_share * len(beats):
        return ibis, len(extra), False
    if rule.neighbour_tol is not None:
        ref = local_median(ibis, rule.context)
        before = np.where(extra >= 2, ibis[np.maximum(extra - 2, 0)], np.nan)
        after = np.where(extra + 1 < len(ibis), ibis[np.minimum(extra + 1, len(ibis) - 1)], np.nan)
        ok = lambda x: ~np.isfinite(x) | (np.abs(x / ref[extra] - 1) <= rule.neighbour_tol)  # noqa: E731
        extra = extra[ok(before) & ok(after)]
        if len(extra) == 0:
            return ibis, 0, False
    keep = np.ones(len(ibis), dtype=bool)
    keep[extra - 1] = False
    keep[extra] = False
    return ibis[keep], len(extra), True
