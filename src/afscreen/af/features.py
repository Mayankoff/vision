"""Step 10 - rhythm-irregularity features from inter-beat intervals (IBIs).

Timing-only features, so a classifier trained on contact PPG transfers to camera pulses
(waveform shape differs between the two; beat timing does not). The set follows Dash et al.
(2009) and McManus et al. (2013).
"""

from __future__ import annotations

import numpy as np

FEATURES = ("nrmssd", "shannon_entropy", "sample_entropy", "sd1_sd2", "tpr")

# Plausible IBIs (s): 30-200 BPM, the heart-rate band of the whole system.
IBI_RANGE = (0.3, 2.0)


def clean_ibis(ibis: np.ndarray, rng: tuple[float, float] = IBI_RANGE) -> np.ndarray:
    """Drop intervals outside the plausible range (missed or doubled beats)."""
    ibis = np.asarray(ibis, dtype=np.float64)
    return ibis[(ibis >= rng[0]) & (ibis <= rng[1])]


def nrmssd(ibis: np.ndarray) -> float:
    """RMSSD divided by the mean IBI: beat-to-beat change, independent of heart rate."""
    return float(np.sqrt(np.mean(np.diff(ibis) ** 2)) / np.mean(ibis))


def shannon_entropy(ibis: np.ndarray, bins: int = 16) -> float:
    """Shannon entropy (bits / log2(bins), 0-1) of the IBI histogram, after dropping the 8
    shortest and 8 longest IBIs as Dash et al. (2009) do (skipped for short windows)."""
    x = np.sort(ibis)
    if len(x) > 32:
        x = x[8:-8]
    if np.ptp(x) == 0:
        return 0.0
    counts, _ = np.histogram(x, bins=bins)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(bins))


def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Sample entropy (Richman & Moorman, 2000) with tolerance ``r`` x std. High = unpredictable.

    Returns the value for one match when no (m+1)-length match exists, so short or very
    irregular windows give a large finite number instead of infinity.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < m + 2 or np.std(x) == 0:
        return 0.0
    tol = r * np.std(x)

    def matches(k: int) -> int:
        t = np.array([x[i:i + k] for i in range(n - m)])
        d = np.max(np.abs(t[:, None, :] - t[None, :, :]), axis=2)
        return int((d <= tol).sum() - len(t))  # exclude self-matches

    b, a = matches(m), matches(m + 1)
    if b == 0:
        return 0.0
    return float(-np.log(max(a, 1) / b))


def sd1_sd2(ibis: np.ndarray) -> float:
    """Poincare SD1 / SD2: short-term over long-term variability. Near 1 for AF (a round cloud),
    small for a normal rhythm (an elongated cloud along the identity line)."""
    a, b = ibis[:-1], ibis[1:]
    sd1 = np.std((b - a) / np.sqrt(2))
    sd2 = np.std((b + a) / np.sqrt(2))
    return float(sd1 / sd2) if sd2 > 0 else 0.0


def tpr(ibis: np.ndarray) -> float:
    """Turning-point ratio: share of IBIs that are a local peak or trough. About 2/3 for random
    intervals (AF); lower for the smooth, breathing-linked changes of a normal rhythm."""
    a, b, c = ibis[:-2], ibis[1:-1], ibis[2:]
    turns = ((b > a) & (b > c)) | ((b < a) & (b < c))
    return float(turns.mean()) if len(turns) else 0.0


def ibi_features(ibis: np.ndarray, min_ibis: int = 12) -> np.ndarray | None:
    """Feature vector in :data:`FEATURES` order, or None when too few plausible IBIs remain."""
    x = clean_ibis(ibis)
    if len(x) < min_ibis:
        return None
    return np.array([nrmssd(x), shannon_entropy(x), sample_entropy(x), sd1_sd2(x), tpr(x)])
