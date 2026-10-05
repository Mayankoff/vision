import numpy as np
import pytest

from rppg.signal.hr import hr_fft, hr_peaks, refine_peaks, windowed_hr

FS = 30.0


@pytest.mark.parametrize("bpm", [45, 60, 72.5, 100, 150, 175])
def test_fft_and_peaks_on_sinusoid(bpm):
    t = np.arange(int(30 * FS)) / FS
    x = np.sin(2 * np.pi * bpm / 60 * t)
    assert hr_fft(x, FS) == pytest.approx(bpm, abs=0.1)
    assert hr_peaks(x, FS) == pytest.approx(bpm, abs=0.5)


def test_fft_ignores_peaks_outside_band():
    t = np.arange(900) / FS
    x = 3 * np.sin(2 * np.pi * 0.3 * t) + np.sin(2 * np.pi * 1.0 * t)  # strong 18 BPM respiration-like component
    assert hr_fft(x, FS) == pytest.approx(60, abs=0.1)


def test_peaks_not_fooled_by_second_bump_per_beat():
    t = np.arange(900) / FS
    phase = 2 * np.pi * 50 / 60 * t
    x = np.sin(phase) + 0.45 * np.sin(2 * phase - 0.8)  # dicrotic-notch-like secondary bump
    assert hr_peaks(x, FS) == pytest.approx(50, abs=1)


def test_refine_peaks_finds_subsample_maximum():
    n = np.arange(20)
    x = -((n - 7.3) ** 2)
    assert refine_peaks(x, np.array([7]))[0] == pytest.approx(7.3, abs=1e-9)


def test_windowed_hr_tracks_changing_rate():
    t = np.arange(int(40 * FS)) / FS
    f = np.where(t < 20, 1.0, 1.5)
    x = np.sin(2 * np.pi * np.cumsum(f) / FS)
    times, hrs = windowed_hr(x, FS, window_s=10, step_s=5)
    assert hrs[0] == pytest.approx(60, abs=1) and hrs[-1] == pytest.approx(90, abs=1)
    assert len(times) == len(hrs)
