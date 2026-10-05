"""Block 2 - face detection and tracking with MediaPipe FaceLandmarker.

Uses the MediaPipe Tasks API (the legacy ``mp.solutions.face_mesh`` API is no
longer shipped in recent mediapipe releases). The model returns 478 landmarks:
the 468 Face Mesh landmarks plus 10 iris landmarks, so Face Mesh landmark
indices from the literature remain valid.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"


def ensure_model(path: str | os.PathLike | None = None) -> Path:
    """Return the path to the FaceLandmarker model, downloading it if missing."""
    path = Path(path) if path else DEFAULT_MODEL_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class FaceLandmarker:
    """Thin wrapper returning landmarks as an (N, 2) pixel array, or None.

    In ``video`` mode MediaPipe tracks the face between frames, which is both
    faster and temporally smoother than independent per-frame detection.
    Timestamps passed to :meth:`detect` must then be strictly increasing.
    """

    def __init__(self, model_path=None, mode: str = "video", min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._mode = mode
        running_mode = {
            "video": vision.RunningMode.VIDEO,
            "image": vision.RunningMode.IMAGE,
        }[mode]
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_s: float | None = None) -> np.ndarray | None:
        """Detect landmarks in an RGB uint8 frame.

        Returns an (478, 2) float array of (x, y) pixel coordinates, or None
        if no face is found.
        """
        frame_rgb = np.ascontiguousarray(frame_rgb)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        if self._mode == "video":
            ts_ms = int(round((timestamp_s or 0.0) * 1000))
            ts_ms = max(ts_ms, self._last_ts_ms + 1)  # must be strictly increasing
            self._last_ts_ms = ts_ms
            result = self._landmarker.detect_for_video(image, ts_ms)
        else:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        h, w = frame_rgb.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=np.float32)
        return pts

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
