import numpy as np
import pytest

from afscreen import synth
from afscreen.signal.beats import beat_agreement, detect_beats, intervals, match_count
from afscreen.signal.filters import bandpass, clean_pulse, detrend, fill_gaps
from afscreen.signal.pos import pos
from afscreen.signal.quality import beat_shape_consistency
from afscreen.signal.spectrum import hr_fft, peak_share

FS = 30.0


def test_fill_gaps_interpolates():
    x = np.array([0.0, np.nan, 2.0, np.nan, 4.0])
    assert np.allclose(fill_gaps(x), [0, 1, 2, 3, 4])


def test_detrend_removes_slow_drift_keeps_pulse():
    t = np.arange(900) / FS
    pulse = np.sin(2 * np.pi * 1.2 * t)
    y = detrend(pulse + 5 * t / t[-1])
    assert np.corrcoef(y, pulse)[0, 1] > 0.99


def test_bandpass_rejects_out_of_band():
    t = np.arange(900) / FS
    y = bandpass(np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 1.5 * t), FS)
    assert np.corrcoef(y[60:-60], np.sin(2 * np.pi * 1.5 * t)[60:-60])[0, 1] > 0.98


@pytest.mark.parametrize("bpm", [38, 72, 150])
def test_pos_recovers_heart_rate_across_the_wide_band(bpm):
    beats = synth.regular_beats(30, bpm, seed=1)
    d = synth.patch_traces(beats, FS, 30, n_patches=1, seed=2)
    pulse = clean_pulse(pos(d["rgb"][:, 0], FS), FS)
    assert hr_fft(pulse, FS) == pytest.approx(60 / np.mean(np.diff(beats)), abs=1.5)


@pytest.mark.parametrize("rhythm", ["regular", "af"])
def test_detect_beats_matches_true_beats(rhythm):
    beats = synth.regular_beats(30, 75, seed=3) if rhythm == "regular" else synth.af_beats(30, 95, seed=3)
    t = np.arange(int(30 * FS)) / FS
    found = detect_beats(bandpass(synth.pulse_from_beats(t, beats), FS), FS)
    inner = beats[(beats > 1) & (beats < 29)]
    assert match_count(found, inner, 0.05) >= len(inner) - 1


def test_match_and_agreement():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert match_count(a, a + 0.03, 0.05) == 4
    assert beat_agreement(a, a + 0.5, 0.1) == 0.0
    assert np.allclose(intervals(a), 1.0)


def test_peak_share_is_low_for_af_but_beat_shape_stays_high():
    """The core design point: a spectral-sharpness score would reject clean AF; beat shape does not."""
    t = np.arange(int(60 * FS)) / FS
    reg = bandpass(synth.pulse_from_beats(t, synth.regular_beats(60, 75, seed=4)), FS)
    af = bandpass(synth.pulse_from_beats(t, synth.af_beats(60, 90, irregularity=0.35, seed=4)), FS)
    assert peak_share(af, FS) < 0.5 * peak_share(reg, FS)
    assert beat_shape_consistency(af, FS) > 0.8
    assert beat_shape_consistency(reg, FS) > 0.8
    noise = bandpass(np.random.default_rng(0).standard_normal(len(t)), FS)
    assert beat_shape_consistency(noise, FS) < beat_shape_consistency(af, FS) - 0.2
