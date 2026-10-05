"""End-to-end classical rPPG pipeline (Blocks 1-8).

Split into two stages so experiments are cheap to repeat:

1. :func:`extract_traces` - the expensive vision part (face landmarks + ROI
   averaging) run once per video, producing per-ROI RGB traces that can be
   cached with :meth:`Traces.save`.
2. :func:`estimate` - the signal part (rPPG method -> pulse -> HR), which runs
   in milliseconds on a cached trace, so every method/ROI/setting can be
   compared on exactly the same input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from rppg.face.roi import SKIN_ROIS, draw_rois, face_box, roi_masks
from rppg.protocol import PROTOCOL
from rppg.signal.hr import estimate_hr, windowed_hr
from rppg.signal.methods import METHODS
from rppg.signal.preprocess import fill_gaps, postprocess_pulse, resample_uniform, spatial_mean

# Traces stored per video. "skin" = union of the three skin ROIs (the default),
# "face_box" = toolbox-style enlarged face box including background/hair.
TRACE_NAMES = (*SKIN_ROIS, "skin", "face_box")


@dataclass
class Traces:
    rgb: dict[str, np.ndarray]  # name -> (T, 3) mean RGB, NaN where no face
    t: np.ndarray  # frame times (s)
    fps: float
    face_found: np.ndarray  # (T,) bool, True where landmarks were detected in that frame
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.t)

    @property
    def detection_rate(self) -> float:
        return float(self.face_found.mean()) if len(self.face_found) else 0.0

    def signal(self, roi: str = "skin", fs: float | None = None) -> tuple[np.ndarray, float]:
        """Gap-filled RGB trace for ``roi`` on a uniform time grid. Returns (rgb, fs)."""
        rgb = fill_gaps(self.rgb[roi])
        fs = fs or self.fps
        dt = np.diff(self.t)
        if len(dt) and np.max(np.abs(dt - 1.0 / fs)) > 0.25 / fs:  # jittery timestamps (e.g. webcam)
            _, rgb = resample_uniform(self.t, rgb, fs)
        return rgb, fs

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            t=self.t,
            fps=self.fps,
            face_found=self.face_found,
            **{f"rgb_{k}": v for k, v in self.rgb.items()},
            **{f"meta_{k}": np.asarray(v) for k, v in self.meta.items()},
        )

    @classmethod
    def load(cls, path) -> "Traces":
        z = np.load(path, allow_pickle=False)
        rgb = {k[4:]: z[k] for k in z.files if k.startswith("rgb_")}
        meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
        meta = {k: (v.item() if v.ndim == 0 else v) for k, v in meta.items()}
        return cls(rgb=rgb, t=z["t"], fps=float(z["fps"]), face_found=z["face_found"], meta=meta)


def extract_traces(
    frames: Iterable[np.ndarray],
    fps: float,
    landmarker,
    timestamps: Iterable[float] | None = None,
    max_hold_s: float = 0.5,
    debug_video: str | None = None,
) -> Traces:
    """Blocks 2-4a: landmarks -> ROI masks -> per-frame mean RGB for every trace in TRACE_NAMES.

    If the face is lost, the last landmarks are reused for up to ``max_hold_s``
    seconds (short dropouts are common under motion); after that the frame is
    marked NaN and later interpolated by :meth:`Traces.signal`.
    """
    rows: dict[str, list] = {k: [] for k in TRACE_NAMES}
    times, found = [], []
    last_pts, last_seen = None, -np.inf
    ts_iter = iter(timestamps) if timestamps is not None else None
    writer = None
    for i, frame in enumerate(frames):
        t = next(ts_iter) if ts_iter is not None else i / fps
        pts = landmarker.detect(frame, t)
        found.append(pts is not None)
        if pts is not None:
            last_pts, last_seen = pts, t
        elif t - last_seen > max_hold_s:
            last_pts = None
        times.append(t)

        if last_pts is None:
            for k in TRACE_NAMES:
                rows[k].append(np.full(3, np.nan))
        else:
            masks = roi_masks(frame.shape, last_pts)
            for k, m in masks.items():
                rows[k].append(spatial_mean(frame, m))
            rows["skin"].append(spatial_mean(frame, np.logical_or.reduce(list(masks.values()))))
            x0, y0, x1, y1 = face_box(frame.shape, last_pts)
            rows["face_box"].append(spatial_mean(frame[y0:y1, x0:x1]))

        if debug_video:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if last_pts is not None:
                bgr = draw_rois(bgr, last_pts)
            if writer is None:
                h, w = bgr.shape[:2]
                writer = cv2.VideoWriter(debug_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(bgr)
    if writer is not None:
        writer.release()
    if not times:
        raise ValueError("no frames to process")
    return Traces(
        rgb={k: np.asarray(v) for k, v in rows.items()},
        t=np.asarray(times, dtype=np.float64),
        fps=float(fps),
        face_found=np.asarray(found, dtype=bool),
    )


@dataclass
class PulseResult:
    bvp: np.ndarray  # post-processed pulse waveform (unit variance)
    fs: float
    hr_fft: float
    hr_peak: float

    @property
    def hr(self) -> float:
        return self.hr_fft if PROTOCOL.hr_method == "fft" else self.hr_peak


def estimate(rgb: np.ndarray, fs: float, method: str = "pos", band=PROTOCOL.hr_band) -> PulseResult:
    """Blocks 5A-7: RGB trace -> rPPG method -> common post-processing -> HR."""
    raw = METHODS[method](np.asarray(rgb, dtype=np.float64), fs, band=band)
    bvp = postprocess_pulse(raw, fs, band)
    return PulseResult(bvp=bvp, fs=fs, hr_fft=estimate_hr(bvp, fs, "fft", band), hr_peak=estimate_hr(bvp, fs, "peak", band))


def reference_hr(bvp_gt: np.ndarray, fs: float, band=PROTOCOL.hr_band) -> tuple[np.ndarray, float, float]:
    """Ground-truth pulse and HR, processed exactly like the predictions."""
    clean = postprocess_pulse(bvp_gt, fs, band)
    return clean, estimate_hr(clean, fs, "fft", band), estimate_hr(clean, fs, "peak", band)


def windowed_pair(pred_bvp: np.ndarray, gt_bvp: np.ndarray, fs: float, window_s: float, step_s: float = PROTOCOL.window_step_s):
    """Per-window predicted and ground-truth HR for the secondary (windowed) table."""
    _, hr_p = windowed_hr(pred_bvp, fs, window_s, step_s)
    _, hr_g = windowed_hr(gt_bvp, fs, window_s, step_s)
    return hr_p, hr_g
