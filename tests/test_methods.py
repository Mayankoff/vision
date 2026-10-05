import numpy as np
import pytest

from rppg.pipeline import estimate
from rppg.signal.methods import METHODS
from rppg.synth import synthetic_rgb_trace

FS = 30.0


@pytest.mark.parametrize("method", sorted(METHODS))
@pytest.mark.parametrize("bpm", [45, 60, 72, 95, 130, 170])
def test_every_method_recovers_hr(method, bpm):
    _, rgb, _ = synthetic_rgb_trace(hr_bpm=bpm, fs=FS, duration_s=30, seed=bpm)
    assert estimate(rgb, FS, method).hr_fft == pytest.approx(bpm, abs=1.0)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_output_length_matches_input(method):
    _, rgb, _ = synthetic_rgb_trace(duration_s=12.3)
    assert estimate(rgb, FS, method).bvp.shape == (rgb.shape[0],)


@pytest.mark.parametrize("bpm", [50, 72, 110])
def test_pos_peak_hr(bpm):
    _, rgb, _ = synthetic_rgb_trace(hr_bpm=bpm, seed=3)
    assert estimate(rgb, FS, "pos").hr_peak == pytest.approx(bpm, abs=2.0)


def test_chrom_and_pos_cancel_intensity_flicker_but_green_does_not():
    """A 108 BPM brightness flicker (5x the pulse amplitude) changes all channels
    proportionally. CHROM and POS project it out; the green channel cannot."""
    _, rgb, _ = synthetic_rgb_trace(hr_bpm=72, flicker_hz=1.8, flicker_strength=0.01, seed=0)
    assert estimate(rgb, FS, "pos").hr_fft == pytest.approx(72, abs=1)
    assert estimate(rgb, FS, "chrom").hr_fft == pytest.approx(72, abs=1)
    assert estimate(rgb, FS, "green").hr_fft == pytest.approx(108, abs=1)


def test_pulse_correlates_with_ground_truth():
    _, rgb, gt = synthetic_rgb_trace(hr_bpm=80, seed=5)
    for m in ("pos", "chrom"):
        bvp = estimate(rgb, FS, m).bvp
        core = slice(60, -60)
        assert abs(np.corrcoef(bvp[core], gt[core])[0, 1]) > 0.8, m
