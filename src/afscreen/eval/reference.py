"""Reference values (heart rate, beats) from a dataset's contact pulse signal, computed with the
same estimators as our predictions so the comparison is fair."""

from __future__ import annotations

import numpy as np

from afscreen.config import SETTINGS
from afscreen.signal.beats import detect_beats
from afscreen.signal.filters import clean_pulse
from afscreen.signal.spectrum import hr_fft


def reference_pulse(bvp: np.ndarray, fs: float) -> np.ndarray:
    return clean_pulse(np.asarray(bvp, dtype=np.float64), fs, SETTINGS.hr_band)


def reference_hr(bvp: np.ndarray, fs: float) -> float:
    return hr_fft(reference_pulse(bvp, fs), fs, SETTINGS.hr_band)


def reference_beats(bvp: np.ndarray, fs: float) -> np.ndarray:
    return detect_beats(reference_pulse(bvp, fs), fs, SETTINGS.hr_band)


def align_reference(ref: np.ndarray, pred: np.ndarray, fs: float, max_lag_s: float = 1.0) -> tuple[np.ndarray, float, int]:
    """Shift and sign-correct a cleaned reference pulse to best match the predicted pulse.

    A finger or ear sensor sees the pulse at a different time than the face (pulse transit
    time), and on real videos the camera pulse is inverted relative to a contact PPG (more blood
    = less reflected light; oximeters plot more blood upwards). Neither changes heart rate or
    beat intervals, but both break beat-time matching. Returns (aligned reference, lag in s,
    sign); a positive lag means the reference was delayed.
    """
    ref, pred = np.asarray(ref, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    max_lag = int(round(max_lag_s * fs))
    best, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        a = ref[max(0, -lag):len(ref) - max(0, lag)]
        b = pred[max(0, lag):len(pred) - max(0, -lag)]
        c = float(np.corrcoef(a, b)[0, 1])
        if abs(c) > abs(best):
            best, best_lag = c, lag
    sign = 1 if best >= 0 else -1
    shifted = np.roll(ref, best_lag)  # wrap-around at the edges only touches |lag| samples
    return sign * shifted, best_lag / fs, sign
