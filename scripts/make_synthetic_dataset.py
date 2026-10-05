"""Build a tiny UBFC-rPPG-format dataset from one face photo, with known heart rates.

    python scripts/make_synthetic_dataset.py face.jpg --out data/synthetic-ubfc

Each "subject" is the same photo with its face pixels pulsing at a different
HR, plus small head jitter and sensor noise. Use it to smoke-test the whole
dataset workflow (extract_traces.py -> run_classical.py) before the real
datasets arrive. It is NOT a benchmark: a still photo has no real motion,
lighting change or physiology.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from rppg.face.landmarker import FaceLandmarker
from rppg.synth import pulse_wave, synthetic_face_video


def face_skin_mask(face_rgb: np.ndarray, model=None) -> np.ndarray:
    with FaceLandmarker(model, mode="image") as lm:
        pts = lm.detect(face_rgb)
    if pts is None:
        raise SystemExit("no face found in the photo")
    mask = np.zeros(face_rgb.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pts[:468]).astype(np.int32)), 1)
    return mask.astype(bool)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for fourcc in ("FFV1", "MJPG"):  # prefer lossless so the tiny pulse is not compressed away
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
    raise SystemExit("no usable video codec")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("face", help="photo of a frontal face")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hrs", default="58,72,88,104", help="comma-separated heart rates, one subject each")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-side", type=int, default=480, help="downscale the photo to this size")
    args = ap.parse_args()

    face = cv2.cvtColor(cv2.imread(args.face), cv2.COLOR_BGR2RGB)
    scale = args.max_side / max(face.shape[:2])
    if scale < 1:
        face = cv2.resize(face, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = face_skin_mask(face)

    for i, hr in enumerate(float(h) for h in args.hrs.split(",")):
        d = Path(args.out) / f"subject{i + 1}"
        d.mkdir(parents=True, exist_ok=True)
        h, w = face.shape[:2]
        writer = open_writer(d / "vid.avi", args.fps, (w, h))
        for frame in synthetic_face_video(face, mask, hr, args.fps, args.duration, seed=i):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        t = np.arange(int(args.duration * args.fps)) / args.fps
        bvp = pulse_wave(t, hr, rng=np.random.default_rng(i))
        with open(d / "ground_truth.txt", "w") as f:  # UBFC DATASET_2 layout: BVP, HR, time
            f.write(" ".join(f"{v:.6e}" for v in bvp) + "\n")
            f.write(" ".join(f"{hr:.6e}" for _ in t) + "\n")
            f.write(" ".join(f"{v:.6e}" for v in t) + "\n")
        print(f"{d}: {hr:.0f} BPM")


if __name__ == "__main__":
    main()
