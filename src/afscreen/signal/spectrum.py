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


def hr_fft_breathing_aware(x: np.ndarray, fs: float, breathing_bpm: float, band=SETTINGS.hr_band,
                           tol_bpm: float = 3.0, n_candidates: int = 5, max_harmonic: int = 3) -> float:
    """Heart rate from the largest spectral peak that is not the breathing rate or a harmonic of it.

    With the band widened down to 30 BPM, slow or deep breathing (and its harmonics) can give a
    larger peak than the pulse. Up to ``n_candidates`` peaks are tried, largest first; a peak
    within ``tol_bpm`` of k x breathing rate (k = 1 .. ``max_harmonic``) is skipped. Higher
    harmonics are negligible, and checking them would reject almost any heart rate (every rate
    is near *some* multiple of the breathing rate). Falls back to the plain FFT heart rate when
    every candidate is skipped or the breathing rate is unknown.
    """
    if not np.isfinite(breathing_bpm) or breathing_bpm <= 0:
        return hr_fft(x, fs, band)
    # Hann window: the plain zero-padded periodogram has sidelobes next to a rejected peak that
    # would otherwise be taken as the next candidate.
    x = np.asarray(x, dtype=np.float64)
    nfft = max(_next_pow2(len(x)), SETTINGS.min_nfft)
    f, p = signal.periodogram(x, fs=fs, nfft=nfft, window="hann", detrend=False)
    m = (f >= band[0]) & (f <= band[1])
    f, p = f[m] * 60.0, p[m]
    idx, _ = signal.find_peaks(p, prominence=0.05 * p.max() if p.max() > 0 else None)
    if len(idx) == 0:
        return hr_fft(x, fs, band)
    harmonics = breathing_bpm * np.arange(1, max_harmonic + 1)
    for i in idx[np.argsort(p[idx])[::-1]][:n_candidates]:
        if np.all(np.abs(f[i] - harmonics) > tol_bpm):
            return float(f[i])
    return hr_fft(x, fs, band)


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
