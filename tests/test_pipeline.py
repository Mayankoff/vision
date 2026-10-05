import numpy as np
import pytest

from rppg.face.roi import ROI_POLYGONS
from rppg.pipeline import TRACE_NAMES, Traces, estimate, extract_traces, reference_hr
from rppg.synth import PULSE_RGB, pulse_wave

FS = 30.0


class FakeLandmarker:
    """Returns fixed landmarks (cheeks/forehead as squares) except on frames listed in ``missing``."""

    def __init__(self, missing=()):
        self.missing, self.i = set(missing), 0
        pts = np.full((478, 2), 50.0)
        for name, (x, y) in {"forehead": (40, 10), "right_cheek": (20, 40), "left_cheek": (60, 40)}.items():
            idx = ROI_POLYGONS[name]
            corners = np.array([[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]], float)
            pts[idx] = corners[np.arange(len(idx)) * 4 // len(idx)]
        pts[0], pts[1] = (15, 5), (75, 55)  # widen the face box
        self.pts = pts

    def detect(self, frame, t):
        i, self.i = self.i, self.i + 1
        return None if i in self.missing else self.pts


def pulsing_frames(bpm, n, strength=0.02):
    p = pulse_wave(np.arange(n) / FS, bpm, rng=0)
    for i in range(n):
        frame = np.full((100, 100, 3), (180.0, 120.0, 100.0))
        frame[5:60, 15:75] *= 1 + strength * p[i] * PULSE_RGB / PULSE_RGB.mean()
        yield np.clip(frame, 0, 255).astype(np.uint8)


def test_extract_and_estimate_recovers_hr():
    tr = extract_traces(pulsing_frames(84, 450), FS, FakeLandmarker())
    assert set(tr.rgb) == set(TRACE_NAMES) and tr.n_frames == 450 and tr.detection_rate == 1.0
    for roi in ("skin", "forehead", "face_box"):
        rgb, fs = tr.signal(roi)
        assert estimate(rgb, fs, "pos").hr_fft == pytest.approx(84, abs=1), roi


def test_short_dropouts_are_held_long_ones_become_nan():
    missing = set(range(10, 13)) | set(range(100, 140))  # 0.1 s and 1.3 s dropouts
    tr = extract_traces(pulsing_frames(70, 200), FS, FakeLandmarker(missing), max_hold_s=0.5)
    skin = tr.rgb["skin"]
    assert not np.isnan(skin[10:13]).any()  # held from the last detection
    assert np.isnan(skin[116:140]).all()  # beyond the 0.5 s hold
    assert not np.isnan(tr.signal("skin")[0]).any()  # gaps are interpolated for processing
    assert tr.detection_rate == pytest.approx(1 - 43 / 200)


def test_traces_roundtrip(tmp_path):
    tr = extract_traces(pulsing_frames(70, 60), FS, FakeLandmarker())
    tr.meta = {"id": "subject1", "subject": "1", "gt_bvp": np.arange(60.0), "label_motion": "Walking"}
    tr.save(tmp_path / "x.npz")
    back = Traces.load(tmp_path / "x.npz")
    assert back.fps == FS and back.meta["id"] == "subject1" and back.meta["label_motion"] == "Walking"
    np.testing.assert_array_equal(back.rgb["skin"], tr.rgb["skin"])
    np.testing.assert_array_equal(back.meta["gt_bvp"], np.arange(60.0))


def test_jittery_timestamps_are_resampled():
    tr = extract_traces(pulsing_frames(70, 300), FS, FakeLandmarker())
    rng = np.random.default_rng(0)
    tr.t = np.cumsum(rng.uniform(0.02, 0.047, tr.n_frames))  # webcam-like jitter around 30 fps
    rgb, fs = tr.signal("skin")
    assert fs == FS and abs(len(rgb) - (tr.t[-1] - tr.t[0]) * FS) <= 1


def test_reference_hr_matches_known_rate():
    bvp = pulse_wave(np.arange(900) / FS, 66, rng=1)
    _, hr_fft, hr_peak = reference_hr(bvp, FS)
    assert hr_fft == pytest.approx(66, abs=0.5) and hr_peak == pytest.approx(66, abs=1)
