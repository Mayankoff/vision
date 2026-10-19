import numpy as np

from afscreen import synth


def test_regular_rhythm_rate_and_small_variation():
    b = synth.regular_beats(60, 70, seed=1)
    ibi = np.diff(b)
    assert abs(60 / ibi.mean() - 70) < 1
    assert ibi.std() < 0.04


def test_af_rhythm_is_irregular():
    ibi = np.diff(synth.af_beats(60, 90, seed=1))
    assert abs(60 / ibi.mean() - 90) < 6
    assert ibi.std() / ibi.mean() > 0.12
    # no pattern: successive differences are as large as the spread itself
    assert np.sqrt(np.mean(np.diff(ibi) ** 2)) > ibi.std()


def test_extra_beats_give_short_then_long_interval():
    b = synth.regular_beats(60, 60, rsa_ms=0, jitter_ms=0, seed=1)
    e = synth.with_extra_beats(b, every=8, seed=1)
    ibi = np.diff(e)
    short = np.flatnonzero(ibi < 0.8)
    assert len(short) >= 5
    assert np.all(ibi[short + 1] > 1.2)


def test_pulse_peaks_at_beats():
    t = np.arange(900) / 30.0
    b = synth.regular_beats(30, 72, seed=2)
    p = synth.pulse_from_beats(t, b)
    assert abs(p.mean()) < 1e-9 and abs(p.std() - 1) < 1e-9
    i = int(round(b[5] * 30))
    assert p[i] >= p[i - 3: i + 4].max() - 0.05
