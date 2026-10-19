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
