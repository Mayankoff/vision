"""Steps 1-3 for a whole video: landmarks and head pose -> fine-patch colour traces.

Traces are the slow part (MediaPipe on every frame), so they are computed once per video and
cached; everything after them (trust, fusion, beats, AF) runs on the cached traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from afscreen.face.patches import N_FINE, measure, merge
from afscreen.face.tracker import FaceTracker


@dataclass
class PatchTraces:
    fps: float
    rgb: np.ndarray  # (T, 25, 3) mean colour per fine patch; NaN where no face
    counts: np.ndarray  # (T, 25) pixels per fine patch
    facing: np.ndarray  # (T, 25) |cos| of the patch normal vs the camera axis
    pose: np.ndarray  # (T, 6) yaw, pitch, roll (deg), x, y, z (cm); NaN where no face
    meta: dict = field(default_factory=dict)
    ref_pulse: np.ndarray | None = None  # (T,) reference contact pulse aligned to the frames, if known
    ref_beats: np.ndarray | None = None  # exact beat times (s), if known (synthetic videos)

    @property
    def n_frames(self) -> int:
        return self.rgb.shape[0]

    @property
    def detection_rate(self) -> float:
        return float(np.isfinite(self.pose[:, 0]).mean()) if self.n_frames else 0.0

    def at_level(self, level: int):
        """(rgb (T, L, 3), counts (T, L), facing (T, L)) merged to ``level`` patches."""
        if level == N_FINE:
            return self.rgb, self.counts, self.facing
        return merge(self.rgb, self.counts, self.facing, level)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fps=self.fps, rgb=self.rgb.astype(np.float32), counts=self.counts.astype(np.float32),
                            facing=self.facing.astype(np.float32), pose=self.pose.astype(np.float32),
                            meta=np.array(repr(self.meta)),
                            **({} if self.ref_pulse is None else {"ref_pulse": self.ref_pulse}),
                            **({} if self.ref_beats is None else {"ref_beats": self.ref_beats}))

    @classmethod
    def load(cls, path: str | Path) -> "PatchTraces":
        import ast

        d = np.load(path, allow_pickle=False)
        return cls(fps=float(d["fps"]), rgb=d["rgb"].astype(np.float64), counts=d["counts"].astype(np.float64),
                   facing=d["facing"].astype(np.float64), pose=d["pose"].astype(np.float64),
                   meta=ast.literal_eval(str(d["meta"])),
                   ref_pulse=d["ref_pulse"] if "ref_pulse" in d else None,
                   ref_beats=d["ref_beats"] if "ref_beats" in d else None)


def extract(frames: Iterable[np.ndarray], fps: float, tracker: FaceTracker | None = None,
            max_frames: int | None = None, on_frame=None) -> PatchTraces:
    """Run the tracker on RGB frames and measure every fine patch.

    ``on_frame(i, frame, face, rgb)`` is called per frame if given (used for debug videos).
    """
    own = tracker is None
    tracker = tracker or FaceTracker()
    rgb, counts, facing, pose = [], [], [], []
    try:
        for i, frame in enumerate(frames):
            if max_frames is not None and i >= max_frames:
                break
            face = tracker.detect(frame, i / fps)
            if face is None:
                rgb.append(np.full((N_FINE, 3), np.nan))
                counts.append(np.zeros(N_FINE))
                facing.append(np.full(N_FINE, np.nan))
                pose.append(np.full(6, np.nan))
            else:
                c, n, f = measure(frame, face.points)
                rgb.append(c)
                counts.append(n)
                facing.append(f)
                pose.append(face.pose)
            if on_frame is not None:
                on_frame(i, frame, face, rgb[-1])
    finally:
        if own:
            tracker.close()
    if not rgb:
        raise ValueError("no frames")
    return PatchTraces(fps=float(fps), rgb=np.array(rgb), counts=np.array(counts), facing=np.array(facing),
                       pose=np.array(pose))
