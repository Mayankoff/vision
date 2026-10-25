"""Dataset readers. Each loader returns :class:`Recording` objects whose frames are streamed lazily.

Expected layouts under ``data/`` (see docs/DATA.md):

    UBFC-rPPG/  subject1/vid.avi, subject1/ground_truth.txt, ...
    PURE/       01-01/01-01/Image*.png, 01-01/01-01.json, ...
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

import cv2
import numpy as np


@dataclass
class Recording:
    """One video with its reference pulse (BVP/PPG) signal."""

    dataset: str
    id: str
    subject: str
    fps: float
    frames: Callable[[], Iterator[np.ndarray]]  # yields RGB uint8 (H, W, 3)
    bvp: np.ndarray
    bvp_t: np.ndarray | None = None  # sample times (s) of the reference, if the dataset gives them
    frame_t: np.ndarray | None = None  # frame times (s), if the dataset gives them
    meta: dict = field(default_factory=dict)

    def aligned_bvp(self, n_frames: int) -> np.ndarray:
        """Reference pulse resampled to one sample per video frame."""
        bvp = np.asarray(self.bvp, dtype=np.float64)
        if self.bvp_t is not None and self.frame_t is not None and len(self.frame_t) >= n_frames:
            return np.interp(self.frame_t[:n_frames], self.bvp_t, bvp)
        if len(bvp) == n_frames:
            return bvp
        return np.interp(np.linspace(0, 1, n_frames), np.linspace(0, 1, len(bvp)), bvp)


def iter_video(path: str) -> Iterator[np.ndarray]:
    """Yield RGB frames of a video file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"video not found: {os.path.abspath(path)}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened() or not cap.grab():
        raise IOError(
            f"OpenCV cannot decode {path}. Phone videos are often HEVC/H.265, which OpenCV usually cannot read: "
            "record in H.264 (iPhone: Settings > Camera > Formats > Most Compatible) or convert, e.g. "
            "ffmpeg -i in.mp4 -c:v libx264 -crf 18 out.mp4"
        )
    cap.release()
    cap = cv2.VideoCapture(path)  # reopen so the first frame is not lost
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def downscale(frames: Iterator[np.ndarray], max_side: int | None) -> Iterator[np.ndarray]:
    """Shrink frames whose longer side exceeds ``max_side`` (phone videos are often 1080p or 4K)."""
    for f in frames:
        h, w = f.shape[:2]
        if max_side and max(h, w) > max_side:
            s = max_side / max(h, w)
            f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        yield f


def video_fps(path: str, default: float = 30.0) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 1 else default


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def read_ubfc2_truth(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """UBFC-rPPG DATASET_2 ``ground_truth.txt``: PPG, HR and time (s) on three lines; one sample per frame."""
    with open(path) as f:
        lines = [ln.split() for ln in f.read().strip().split("\n")]
    bvp = np.array(lines[0], dtype=np.float64)
    t = np.array(lines[2], dtype=np.float64) if len(lines) > 2 and len(lines[2]) == len(bvp) else None
    return bvp, t


def read_ubfc1_truth(path: str) -> tuple[np.ndarray, np.ndarray]:
    """UBFC-rPPG DATASET_1 ``gtdump.xmp``: CSV rows of time (ms), HR, SpO2, PPG from a ~60 Hz oximeter.

    Returns the PPG and its sample times (s). The videos have no frame timestamps, so the PPG is
    later stretched over the video (:meth:`Recording.aligned_bvp`).
    """
    d = np.loadtxt(path, delimiter=",", ndmin=2)
    return d[:, 3], (d[:, 0] - d[0, 0]) / 1000.0


def load_ubfc(root: str) -> list[Recording]:
    """UBFC-rPPG DATASET_2: ``subject<N>/vid.avi`` + ``ground_truth.txt``."""
    recs = []
    for d in sorted(glob.glob(os.path.join(root, "subject*")), key=_natural_key):
        vid, gt_file = os.path.join(d, "vid.avi"), os.path.join(d, "ground_truth.txt")
        if not (os.path.exists(vid) and os.path.exists(gt_file)):
            continue
        bvp, t = read_ubfc2_truth(gt_file)
        name = os.path.basename(d)
        recs.append(Recording(dataset="UBFC-rPPG", id=name, subject=re.search(r"\d+", name).group(0),
                              fps=video_fps(vid), frames=lambda vid=vid: iter_video(vid), bvp=bvp,
                              bvp_t=t, frame_t=t))  # one GT sample per frame
    return recs


def load_ubfc1(root: str) -> list[Recording]:
    """UBFC-rPPG DATASET_1 (the earlier, smaller set): ``<N>-gt/vid.avi`` + ``gtdump.xmp``."""
    recs = []
    for d in sorted(glob.glob(os.path.join(root, "*")), key=_natural_key):
        vid, gt_file = os.path.join(d, "vid.avi"), os.path.join(d, "gtdump.xmp")
        if not (os.path.exists(vid) and os.path.exists(gt_file)):
            continue
        bvp, _ = read_ubfc1_truth(gt_file)
        name = os.path.basename(d)
        recs.append(Recording(dataset="UBFC-rPPG-1", id=f"ubfc1_{name}", subject=f"ubfc1_{name}",
                              fps=video_fps(vid), frames=lambda vid=vid: iter_video(vid), bvp=bvp))
    return recs


def _iter_pngs(paths: list[str]) -> Iterator[np.ndarray]:
    for p in paths:
        yield cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def load_pure(root: str) -> list[Recording]:
    """PURE: PNG frames named ``Image<ns timestamp>.png``; the JSON holds the ~60 Hz oximeter
    waveform with its own timestamps, so the reference is interpolated onto the frame times.
    Sessions: 01 steady, 02 talking, 03 slow / 04 fast translation, 05 small / 06 medium rotation."""
    recs = []
    for d in sorted(glob.glob(os.path.join(root, "*-*"))):
        name = os.path.basename(d)
        json_file = os.path.join(d, f"{name}.json")
        pngs = sorted(glob.glob(os.path.join(d, name, "*.png")))
        if not (os.path.exists(json_file) and pngs):
            continue
        with open(json_file) as f:
            data = json.load(f)
        pkg = data["/FullPackage"]
        bvp = np.array([p["Value"]["waveform"] for p in pkg], dtype=np.float64)
        bvp_t = np.array([p["Timestamp"] for p in pkg], dtype=np.float64)
        img_t = np.array([p["Timestamp"] for p in data.get("/Image", [])], dtype=np.float64)
        if len(img_t) != len(pngs):  # fall back to the timestamps in the file names
            img_t = np.array([float(re.search(r"(\d+)", os.path.basename(p)).group(1)) for p in pngs])
        t0 = img_t[0]
        recs.append(Recording(dataset="PURE", id=name, subject=name.split("-")[0], fps=30.0,
                              frames=lambda pngs=pngs: _iter_pngs(pngs), bvp=bvp, bvp_t=(bvp_t - t0) / 1e9,
                              frame_t=(img_t - t0) / 1e9, meta={"session": name.split("-")[1]}))
    return recs


def load_synthetic(root: str) -> list[Recording]:
    """Synthetic videos made by scripts/make_synthetic_videos.py (``<id>/vid.avi`` + ``<id>/truth.npz``)."""
    recs = []
    for d in sorted(glob.glob(os.path.join(root, "*")), key=_natural_key):
        vid, truth = os.path.join(d, "vid.avi"), os.path.join(d, "truth.npz")
        if not (os.path.exists(vid) and os.path.exists(truth)):
            continue
        z = np.load(truth)
        name = os.path.basename(d)
        recs.append(Recording(dataset="synthetic", id=name, subject=name, fps=float(z["fps"]),
                              frames=lambda vid=vid: iter_video(vid), bvp=z["pulse"],
                              meta={"beats": z["beats"].tolist(), "rhythm": str(z["rhythm"])}))
    return recs


LOADERS = {"ubfc": load_ubfc, "ubfc1": load_ubfc1, "pure": load_pure, "synthetic": load_synthetic}
