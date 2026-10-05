"""Block 1 - dataset readers for UBFC-rPPG, PURE and (mini-)MMPD.

Each loader returns a list of :class:`Recording` objects. Frames are streamed
lazily (a 2-minute 640x480 video is ~3 GB as a uint8 array), and the reference
BVP is aligned to the video frames on request.

Expected directory layouts are the same as rPPG-Toolbox's, so one copy of the
raw data serves both codebases:

    UBFC-rPPG/  subject1/vid.avi, subject1/ground_truth.txt, ...
    PURE/       01-01/01-01/Image*.png, 01-01/01-01.json, ...
    MMPD/       subject1/p1_0.mat, subject1/p1_1.mat, ...
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Callable, Iterator

import cv2
import numpy as np


class Recording:
    """One video with its reference pulse signal.

    ``frames()`` yields RGB uint8 (H, W, 3) frames. ``bvp`` and ``meta`` may be
    given directly or produced on first access by ``lazy()`` (used for MMPD,
    where labels and video live in the same large ``.mat`` file).
    """

    def __init__(
        self,
        dataset: str,
        id: str,
        subject: str,
        fps: float,
        frames: Callable[[], Iterator[np.ndarray]],
        bvp: np.ndarray | None = None,
        bvp_t: np.ndarray | None = None,
        frame_t: np.ndarray | None = None,
        meta: dict | None = None,
        lazy: Callable[[], tuple[np.ndarray, dict]] | None = None,
    ):
        self.dataset, self.id, self.subject, self.fps, self.frames = dataset, id, subject, fps, frames
        self.bvp_t, self.frame_t = bvp_t, frame_t  # sample times in seconds, if the dataset provides them
        self._bvp, self._meta, self._lazy = bvp, meta, lazy

    def _load(self) -> None:
        if self._bvp is None and self._lazy is not None:
            self._bvp, self._meta = self._lazy()

    @property
    def bvp(self) -> np.ndarray:
        self._load()
        return self._bvp

    @property
    def meta(self) -> dict:
        self._load()
        return self._meta or {}

    def __repr__(self) -> str:
        return f"Recording({self.dataset}/{self.id}, subject={self.subject}, fps={self.fps})"

    def aligned_bvp(self, n_frames: int) -> np.ndarray:
        """Reference BVP resampled to one sample per video frame."""
        bvp = np.asarray(self.bvp, dtype=np.float64)
        if self.bvp_t is not None and self.frame_t is not None and len(self.frame_t) >= n_frames:
            return np.interp(self.frame_t[:n_frames], self.bvp_t, bvp)
        if len(bvp) == n_frames:
            return bvp
        return np.interp(np.linspace(0, 1, n_frames), np.linspace(0, 1, len(bvp)), bvp)


def iter_video(path: str) -> Iterator[np.ndarray]:
    """Yield RGB frames of a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def downscale(frames: Iterator[np.ndarray], max_side: int | None) -> Iterator[np.ndarray]:
    """Shrink frames whose longer side exceeds ``max_side`` (phone videos are often 1080p/4K).

    ROI averaging is unaffected by resolution, and MediaPipe is much faster on
    smaller frames.
    """
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


# --------------------------------------------------------------------- UBFC-rPPG


def load_ubfc(root: str) -> list[Recording]:
    """UBFC-rPPG DATASET_2. ``ground_truth.txt`` holds BVP, HR and time (s) on three lines."""
    recs = []
    for d in sorted(glob.glob(os.path.join(root, "subject*")), key=_natural_key):
        vid = os.path.join(d, "vid.avi")
        gt_file = os.path.join(d, "ground_truth.txt")
        if not (os.path.exists(vid) and os.path.exists(gt_file)):
            continue
        with open(gt_file) as f:
            lines = [ln.split() for ln in f.read().strip().split("\n")]
        bvp = np.array(lines[0], dtype=np.float64)
        bvp_t = np.array(lines[2], dtype=np.float64) if len(lines) > 2 and len(lines[2]) == len(bvp) else None
        name = os.path.basename(d)
        subject = re.search(r"\d+", name).group(0)
        recs.append(
            Recording(
                dataset="UBFC-rPPG",
                id=name,
                subject=subject,
                fps=video_fps(vid),
                frames=lambda vid=vid: iter_video(vid),
                bvp=bvp,
                # UBFC's GT timestamps are one per frame; frame times are therefore the same grid.
                bvp_t=bvp_t,
                frame_t=bvp_t,
            )
        )
    return recs


# --------------------------------------------------------------------------- PURE


def _iter_pngs(paths: list[str]) -> Iterator[np.ndarray]:
    for p in paths:
        yield cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def load_pure(root: str) -> list[Recording]:
    """PURE. Frames are PNGs named ``Image<ns timestamp>.png``; the JSON holds the
    ~60 Hz pulse-oximeter waveform with its own timestamps, so BVP is
    interpolated onto the frame times."""
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
        recs.append(
            Recording(
                dataset="PURE",
                id=name,
                subject=name.split("-")[0],
                fps=30.0,
                frames=lambda pngs=pngs: _iter_pngs(pngs),
                bvp=bvp,
                bvp_t=(bvp_t - t0) / 1e9,
                frame_t=(img_t - t0) / 1e9,
                meta={"session": name.split("-")[1]},
            )
        )
    return recs


# -------------------------------------------------------------------------- MMPD

MMPD_FIELDS = ("light", "motion", "exercise", "skin_color", "gender", "glasser", "hair_cover", "makeup")

# Integer codes used by rPPG-Toolbox for MMPD labels (its file names look like
# subject5_L1_MO2_E2_S3_GE1_GL2_H2_MA2). Index = code; used to match videos
# between the two codebases and to give labels one canonical spelling.
MMPD_CODES = {
    "light": {"LED-low": 1, "LED-high": 2, "Incandescent": 3, "Nature": 4},
    "motion": {"Stationary": 1, "Stationary (after exercise)": 1, "Rotation": 2, "Talking": 3, "Walking": 4,
               "Watching Videos": 4},  # 'Watching Videos' is an old mislabel of 'Walking'
    "exercise": {"True": 1, "False": 2},
    "gender": {"male": 1, "female": 2},
    "glasser": {"True": 1, "False": 2},
    "hair_cover": {"True": 1, "False": 2},
    "makeup": {"True": 1, "False": 2},
}
_TOOLBOX_TAGS = {"L": "light", "MO": "motion", "E": "exercise", "S": "skin_color", "GE": "gender", "GL": "glasser",
                 "H": "hair_cover", "MA": "makeup"}


def mmpd_canonical_labels(meta: dict) -> dict[str, str]:
    """Labels with one spelling per condition (e.g. 'Stationary (after exercise)' -> 'Stationary')."""
    out = {}
    for field in MMPD_FIELDS:
        if field not in meta:
            continue
        value = str(meta[field])
        if field in MMPD_CODES:
            code = MMPD_CODES[field][value]
            value = next(k for k, v in MMPD_CODES[field].items() if v == code)
        out[field] = value
    return out


def parse_toolbox_mmpd_name(name: str) -> tuple[str, dict[str, str]]:
    """'subject5_L1_MO2_E2_S3_GE1_GL2_H2_MA2' -> ('5', canonical labels)."""
    parts = name.split("_")
    subject = re.search(r"\d+", parts[0]).group(0)
    labels = {}
    for part in parts[1:]:
        tag, code = re.match(r"([A-Z]+)(\d+)", part).groups()
        field = _TOOLBOX_TAGS[tag]
        if field == "skin_color":
            labels[field] = code
        else:
            labels[field] = next(k for k, v in MMPD_CODES[field].items() if v == int(code))
    return subject, labels


def _mat_str(v) -> str:
    v = np.asarray(v).squeeze()
    return str(v.item() if v.size == 1 else v).strip()


def _to_uint8(video: np.ndarray) -> np.ndarray:
    if video.dtype == np.uint8:
        return video
    v = video.astype(np.float32)
    if v.max() <= 1.0 + 1e-6:
        v = v * 255.0
    return np.clip(np.round(v), 0, 255).astype(np.uint8)


def load_mmpd(root: str) -> list[Recording]:
    """MMPD / mini-MMPD ``.mat`` files with ``video``, ``GT_ppg`` and condition labels.

    BVP and condition labels are read on first access, without the video.
    """
    recs = []
    for d in sorted(glob.glob(os.path.join(root, "subject*")), key=_natural_key):
        subject = re.search(r"\d+", os.path.basename(d)).group(0)
        for mat in sorted(glob.glob(os.path.join(d, "*.mat")), key=_natural_key):
            recs.append(_mmpd_recording(mat, subject))
    return recs


def _mmpd_recording(mat_path: str, subject: str) -> Recording:
    from scipy.io import loadmat

    def labels() -> tuple[np.ndarray, dict]:
        m = loadmat(mat_path, variable_names=["GT_ppg", *MMPD_FIELDS])
        bvp = np.asarray(m["GT_ppg"], dtype=np.float64).reshape(-1)
        return bvp, {k: _mat_str(m[k]) for k in MMPD_FIELDS if k in m}

    def frames():
        # Not cached: each video is loaded only while it is being processed.
        for f in loadmat(mat_path, variable_names=["video"])["video"]:
            yield _to_uint8(f)

    return Recording(
        dataset="MMPD",
        id=os.path.splitext(os.path.basename(mat_path))[0],
        subject=subject,
        fps=30.0,
        frames=frames,
        lazy=labels,
    )


LOADERS = {"ubfc": load_ubfc, "pure": load_pure, "mmpd": load_mmpd}
