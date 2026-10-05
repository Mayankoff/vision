"""Block 3 - skin ROI extraction (forehead and cheeks) from face landmarks.

Polygons are MediaPipe Face Mesh landmark indices. "left"/"right" are from the
subject's point of view (``right_cheek`` appears on the left of the image).
The polygons stay inside skin and avoid eyes, eyebrows, nostrils and lips;
the top edge of the forehead polygon may touch the hairline on subjects with a
low fringe.
"""

from __future__ import annotations

import cv2
import numpy as np

ROI_POLYGONS: dict[str, list[int]] = {
    "forehead": [67, 109, 10, 338, 297, 334, 296, 336, 9, 107, 66, 105],
    "right_cheek": [117, 118, 119, 100, 142, 203, 206, 207, 187, 123, 116],
    "left_cheek": [346, 347, 348, 329, 371, 423, 426, 427, 411, 352, 345],
}
SKIN_ROIS = tuple(ROI_POLYGONS)

# Toolbox-style ROI: the landmark bounding box enlarged by this factor
# (rPPG-Toolbox uses LARGE_BOX_COEF = 1.5). Used for the Phase 2 ROI ablation.
FACE_BOX_SCALE = 1.5


def polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    """Boolean mask of the filled polygon ``points`` (K, 2) for an image of ``shape`` (H, W)."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask.astype(bool)


def roi_masks(shape: tuple[int, int], landmarks: np.ndarray, rois=SKIN_ROIS) -> dict[str, np.ndarray]:
    """Boolean masks for each named ROI polygon."""
    return {name: polygon_mask(shape, landmarks[ROI_POLYGONS[name]]) for name in rois}


def face_box(shape: tuple[int, int], landmarks: np.ndarray, scale: float = FACE_BOX_SCALE) -> tuple[int, int, int, int]:
    """Square box (x0, y0, x1, y1) around the landmarks, enlarged by ``scale``, clipped to the image."""
    h, w = shape[:2]
    (x0, y0), (x1, y1) = landmarks.min(axis=0), landmarks.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * scale / 2
    return (
        int(max(0, cx - half)),
        int(max(0, cy - half)),
        int(min(w, cx + half)),
        int(min(h, cy + half)),
    )


def draw_rois(frame_bgr: np.ndarray, landmarks: np.ndarray, rois=SKIN_ROIS) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with ROI outlines drawn (for debug videos and figures)."""
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (0, 255, 0)]
    out = frame_bgr.copy()
    for color, name in zip(colors, rois):
        pts = np.round(landmarks[ROI_POLYGONS[name]]).astype(np.int32)
        cv2.polylines(out, [pts], True, color, 2)
    return out
