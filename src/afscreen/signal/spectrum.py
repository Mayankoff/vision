"""Spectra and FFT heart rate."""

from __future__ import annotations

import numpy as np
from scipy import signal

from afscreen.config import SETTINGS


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


def power_spectrum(x: np.ndarray, fs: float, min_nfft: int = SETTINGS.min_nfft) -> tuple[np.ndarray, np.ndarray]:
    """Zero-padded periodogram (frequencies in Hz, power)."""
    nfft = max(_next_pow2(len(x)), min_nfft)
    return signal.periodogram(np.asarray(x, dtype=np.float64), fs=fs, nfft=nfft, detrend=False)


def hr_fft(x: np.ndarray, fs: float, band=SETTINGS.hr_band) -> float:
    """Heart rate in BPM from the largest spectral peak inside ``band``."""
    f, p = power_spectrum(x, fs)
    mask = (f >= band[0]) & (f <= band[1])
    return float(f[mask][np.argmax(p[mask])] * 60.0)


def peak_share(x: np.ndarray, fs: float, band=SETTINGS.hr_band, tol_bpm: float = 6.0) -> float:
    """Share of in-band power within +/- ``tol_bpm`` of the largest peak (0-1).

    High for a clean, *regular* pulse. Not used for trust or confidence, because an irregular
    rhythm (AF) spreads its power and would score low; kept for the comparison in Phase 5.
    """
    f, p = power_spectrum(x, fs)
    inb = (f >= band[0]) & (f <= band[1])
    f, p = f[inb], p[inb]
    total = p.sum()
    if total <= 0:
        return 0.0
    f0 = f[np.argmax(p)]
    return float(p[np.abs(f - f0) <= tol_bpm / 60.0].sum() / total)
