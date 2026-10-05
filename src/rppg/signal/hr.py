"""Block 7 - heart-rate estimation from a pulse waveform."""

from __future__ import annotations

import numpy as np
from scipy import signal

from rppg.protocol import PROTOCOL


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


def power_spectrum(x: np.ndarray, fs: float, min_nfft: int = PROTOCOL.min_nfft) -> tuple[np.ndarray, np.ndarray]:
    """Zero-padded periodogram (frequencies in Hz, power)."""
    nfft = max(_next_pow2(len(x)), min_nfft)
    return signal.periodogram(np.asarray(x, dtype=np.float64), fs=fs, nfft=nfft, detrend=False)


def hr_fft(x: np.ndarray, fs: float, band=PROTOCOL.hr_band) -> float:
    """HR in BPM from the largest spectral peak inside ``band``."""
    f, p = power_spectrum(x, fs)
    mask = (f >= band[0]) & (f <= band[1])
    return float(f[mask][np.argmax(p[mask])] * 60.0)


def hr_peaks(x: np.ndarray, fs: float, band=PROTOCOL.hr_band) -> float:
    """HR in BPM from beat-to-beat intervals (60 / median IBI).

    Peaks closer than 60% of the dominant (FFT) period are suppressed, so the
    secondary bump of each beat (dicrotic notch / 2nd harmonic) is not counted
    as an extra beat. The median IBI is used instead of the mean so that one
    missed or extra beat does not dominate the estimate.
    """
    x = np.asarray(x, dtype=np.float64)
    dominant_period = 60.0 / hr_fft(x, fs, band) * fs
    min_distance = max(1, int(max(fs / band[1], 0.6 * dominant_period)))
    peaks, _ = signal.find_peaks(x, distance=min_distance, prominence=0.3 * np.std(x))
    if len(peaks) < 2:
        return float("nan")
    return float(60.0 / (np.median(np.diff(refine_peaks(x, peaks))) / fs))


def refine_peaks(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Sub-sample peak positions by fitting a parabola through each peak and its neighbours.

    Without this, beat intervals are quantised to whole frames (at 30 fps and
    72 BPM, one frame is ~3 BPM).
    """
    p = peaks[(peaks > 0) & (peaks < len(x) - 1)]
    a, b, c = x[p - 1], x[p], x[p + 1]
    denom = a - 2 * b + c
    offset = np.where(denom != 0, 0.5 * (a - c) / np.where(denom != 0, denom, 1), 0.0)
    return p + np.clip(offset, -0.5, 0.5)


HR_ESTIMATORS = {"fft": hr_fft, "peak": hr_peaks}


def estimate_hr(x: np.ndarray, fs: float, method: str = PROTOCOL.hr_method, band=PROTOCOL.hr_band) -> float:
    return HR_ESTIMATORS[method](x, fs, band)


def windowed_hr(
    x: np.ndarray,
    fs: float,
    window_s: float,
    step_s: float = PROTOCOL.window_step_s,
    method: str = PROTOCOL.hr_method,
    band=PROTOCOL.hr_band,
) -> tuple[np.ndarray, np.ndarray]:
    """HR per sliding window. Returns (window centre times in s, HR in BPM)."""
    win, step = int(round(window_s * fs)), max(1, int(round(step_s * fs)))
    if len(x) < win:
        return np.array([len(x) / fs / 2]), np.array([estimate_hr(x, fs, method, band)])
    starts = range(0, len(x) - win + 1, step)
    times = np.array([(s + win / 2) / fs for s in starts])
    hrs = np.array([estimate_hr(x[s : s + win], fs, method, band) for s in starts])
    return times, hrs
