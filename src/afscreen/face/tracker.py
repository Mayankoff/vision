"""Step 1 - face landmarks and head pose with MediaPipe FaceLandmarker.

Uses the MediaPipe Tasks API (the legacy ``mp.solutions.face_mesh`` is not shipped in recent
releases). The model gives 478 landmarks (the 468 Face Mesh points plus 10 iris points), each
with a relative depth, and a facial transformation matrix from which the head pose is read.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"


def ensure_model(path: str | os.PathLike | None = None) -> Path:
    """Path to the FaceLandmarker model, downloaded on first use (3.7 MB)."""
    path = Path(path) if path else DEFAULT_MODEL_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


@dataclass
class FaceFrame:
    points: np.ndarray  # (478, 3) x, y in pixels; z (relative depth) in the same pixel scale as x
    pose: np.ndarray  # (6,) yaw, pitch, roll in degrees; x, y, z translation in cm


def pose_from_matrix(m: np.ndarray) -> np.ndarray:
    """(yaw, pitch, roll) in degrees and (x, y, z) translation from a 4x4 transformation matrix."""
    from scipy.spatial.transform import Rotation

    m = np.asarray(m, dtype=np.float64)
    r = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0, keepdims=True)  # remove any scale
    pitch, yaw, roll = Rotation.from_matrix(r).as_euler("xyz", degrees=True)
    return np.array([yaw, pitch, roll, *m[:3, 3]])


class FaceTracker:
    """MediaPipe FaceLandmarker returning :class:`FaceFrame` objects (or None when no face).

    In ``video`` mode MediaPipe tracks the face between frames (faster and smoother than
    detecting in every frame); timestamps must then increase.
    """

    def __init__(self, model_path=None, mode: str = "video", min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._mode = mode
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode={"video": vision.RunningMode.VIDEO, "image": vision.RunningMode.IMAGE}[mode],
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_s: float = 0.0) -> FaceFrame | None:
        frame_rgb = np.ascontiguousarray(frame_rgb)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        if self._mode == "video":
            ts_ms = max(int(round(timestamp_s * 1000)), self._last_ts_ms + 1)
            self._last_ts_ms = ts_ms
            result = self._landmarker.detect_for_video(image, ts_ms)
        else:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        h, w = frame_rgb.shape[:2]
        pts = np.array([(p.x * w, p.y * h, p.z * w) for p in result.face_landmarks[0]], dtype=np.float64)
        mats = result.facial_transformation_matrixes
        pose = pose_from_matrix(mats[0]) if mats else np.full(6, np.nan)
        return FaceFrame(points=pts, pose=pose)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
