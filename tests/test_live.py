from pathlib import Path

import numpy as np
import pytest

from afscreen import synth
from afscreen.live import LiveMonitor, resample

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "results/phase3/mimic/af_classifier.joblib"
EXTRA = ROOT / "results/phase4/mitbih"


def test_resample_irregular_times_keeps_gaps():
    t = np.array([0.0, 0.03, 0.07, 0.1, 0.13, 0.2])
    x = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 6.0])[:, None]
    grid, y = resample(t, x, 30.0)
    assert np.allclose(grid[:3], [0, 1 / 30, 2 / 30])
    assert np.isnan(y[2, 0])  # nearest sample to 0.067 s had no face
    assert abs(y[1, 0] - (1 + 2 * (1 / 30 - 0.03) / 0.07)) < 1e-9  # between the valid samples at 0.03 and 0.1 s


def _feed(mon, beats, duration=40.0, fps=30.0, jitter_s=0.004, seed=0):
    d = synth.patch_traces(beats, fps, duration, seed=seed)
    rng = np.random.default_rng(seed)
    t = np.arange(d["rgb"].shape[0]) / fps + rng.uniform(-jitter_s, jitter_s, d["rgb"].shape[0])  # webcam timing
    for i in range(len(t)):
        mon.add(t[i], d["rgb"][i], np.full(25, 500.0), d["facing"][i], d["pose"][i])


@pytest.mark.skipif(not MODEL.exists(), reason="needs the trained AF model")
@pytest.mark.parametrize("rhythm", ["regular", "af"])
def test_monitor_reports_heart_rate_and_rhythm(rhythm):
    mon = LiveMonitor(model_path=MODEL, extra_beats_dir=EXTRA)
    beats = synth.regular_beats(40, 72, seed=1) if rhythm == "regular" else synth.af_beats(40, 90, seed=1)
    _feed(mon, beats)
    st = mon.update()
    true_hr = 60 / np.mean(np.diff(beats[beats > 32]))
    assert st.face and st.level == 25 and st.weights is not None
    assert abs(st.hr_bpm - true_hr) < (3 if rhythm == "regular" else 12)
    assert st.rhythm == ("regular rhythm" if rhythm == "regular" else "possible AF - see a doctor for an ECG")
    assert st.hr_method.startswith("spectrum" if rhythm == "regular" else "beats")


def test_monitor_waits_for_enough_signal_and_without_a_face():
    mon = LiveMonitor(model_path="missing", extra_beats_dir="missing")
    for i in range(60):  # 2 s
        mon.add(i / 30, np.full((25, 3), np.nan), np.zeros(25), np.full(25, np.nan), np.full(6, np.nan))
    st = mon.update()
    assert not st.face and st.rhythm == "no face" and np.isnan(st.hr_bpm)
