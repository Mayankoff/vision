"""End-to-end test through real MediaPipe on a synthetic pulsing face video.

Opt-in: needs a frontal face photo, e.g.
    RPPG_FACE_IMAGE=path/to/face.jpg pytest -m slow
"""

import os

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.slow
FACE = os.environ.get("RPPG_FACE_IMAGE")


@pytest.mark.skipif(not FACE, reason="set RPPG_FACE_IMAGE to a face photo to run")
@pytest.mark.parametrize("bpm", [62, 96])
def test_mediapipe_pipeline_recovers_hr(bpm):
    from make_synthetic_dataset import face_skin_mask

    from rppg.face.landmarker import FaceLandmarker
    from rppg.pipeline import estimate, extract_traces
    from rppg.synth import synthetic_face_video

    face = cv2.cvtColor(cv2.imread(FACE), cv2.COLOR_BGR2RGB)
    scale = 360 / max(face.shape[:2])
    face = cv2.resize(face, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = face_skin_mask(face)
    with FaceLandmarker() as lm:
        tr = extract_traces(synthetic_face_video(face, mask, bpm, 30.0, 15.0, seed=bpm), 30.0, lm)
    assert tr.detection_rate > 0.95
    for method in ("pos", "chrom"):
        rgb, fs = tr.signal("skin")
        assert estimate(rgb, fs, method).hr_fft == pytest.approx(bpm, abs=1.5), method
