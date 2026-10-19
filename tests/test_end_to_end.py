"""End-to-end through real MediaPipe on synthetic pulsing-face videos.

Opt-in: needs a frontal face photo, e.g.
    AFSCREEN_FACE_IMAGE=path/to/face.png pytest -m slow
"""

import os

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.slow
FACE = os.environ.get("AFSCREEN_FACE_IMAGE")


@pytest.fixture(scope="module")
def face():
    img = cv2.cvtColor(cv2.imread(FACE), cv2.COLOR_BGR2RGB)
    s = 400 / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    from afscreen.synth import skin_mask_from_photo

    return img, skin_mask_from_photo(img)


@pytest.mark.skipif(not FACE, reason="set AFSCREEN_FACE_IMAGE to a face photo to run")
@pytest.mark.parametrize("rhythm", ["regular", "af"])
def test_pipeline_recovers_heart_rate_and_beats(face, rhythm):
    from afscreen import synth
    from afscreen.eval.metrics import ibi_error_ms
    from afscreen.pipeline import analyse
    from afscreen.traces import extract

    img, mask = face
    beats = synth.regular_beats(20, 72, seed=1) if rhythm == "regular" else synth.af_beats(20, 90, seed=1)
    tr = extract(synth.face_video(img, mask, beats, 30.0, 20.0, seed=1), 30.0)
    assert tr.detection_rate > 0.95
    r = analyse(tr, "full")
    true_hr = 60 / np.mean(np.diff(beats))
    assert r.hr_beats_bpm == pytest.approx(true_hr, abs=2.0)
    if rhythm == "regular":  # an irregular rhythm has no single spectral peak
        assert r.hr_bpm == pytest.approx(true_hr, abs=2.0)
    assert ibi_error_ms(r.beats, beats) < 25
    assert r.mean_confidence > 0.6
