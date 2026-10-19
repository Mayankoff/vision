"""Build a small dataset of synthetic face videos with known beat times, from one face photo.

    python scripts/make_synthetic_videos.py face.png --out data/synthetic

Each video is the same photo with its skin pulsing beat by beat; ``truth.npz`` stores the exact
beat times. The set covers the rhythms and conditions of the plan: regular, AF-like, slow, fast,
extra beats, small faces and side-to-side head motion under tinted light. Use it to test the
whole pipeline before the real datasets arrive. It is NOT a benchmark: a still photo has no
real physiology, motion or lighting change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from afscreen import synth

CASES = {
    # name: (rhythm, rhythm kwargs, video kwargs)
    "regular_72": ("regular", dict(hr_bpm=72), {}),
    "regular_64_small": ("regular", dict(hr_bpm=64), dict(scale=0.6)),
    "slow_42": ("regular", dict(hr_bpm=42), {}),
    "fast_128": ("regular", dict(hr_bpm=128), {}),
    "af_88": ("af", dict(mean_hr_bpm=88), {}),
    "af_110": ("af", dict(mean_hr_bpm=110, irregularity=0.35), {}),
    "extra_beats_70": ("extra", dict(hr_bpm=70), {}),
    "regular_76_sway": ("regular", dict(hr_bpm=76), dict(sway_hz=0.9)),
    "af_92_sway": ("af", dict(mean_hr_bpm=92), dict(sway_hz=1.1)),
}


def beats_for(rhythm: str, kwargs: dict, duration: float, seed: int) -> np.ndarray:
    if rhythm == "af":
        return synth.af_beats(duration, seed=seed, **kwargs)
    b = synth.regular_beats(duration, seed=seed, **kwargs)
    return synth.with_extra_beats(b, seed=seed) if rhythm == "extra" else b


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for fourcc in ("FFV1", "MJPG"):  # lossless first, so the tiny pulse is not compressed away
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
    raise SystemExit("no usable video codec")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("face", help="photo of a frontal face")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-side", type=int, default=480)
    ap.add_argument("--cases", default="all", help="comma-separated case names, or 'all'")
    args = ap.parse_args()

    face = cv2.cvtColor(cv2.imread(args.face), cv2.COLOR_BGR2RGB)
    s = args.max_side / max(face.shape[:2])
    if s < 1:
        face = cv2.resize(face, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = synth.skin_mask_from_photo(face)
    names = list(CASES) if args.cases == "all" else args.cases.split(",")
    for i, name in enumerate(names):
        rhythm, rk, vk = CASES[name]
        beats = beats_for(rhythm, rk, args.duration, seed=i + 1)
        d = Path(args.out) / name
        d.mkdir(parents=True, exist_ok=True)
        writer = open_writer(d / "vid.avi", args.fps, (face.shape[1], face.shape[0]))
        for frame in synth.face_video(face, mask, beats, args.fps, args.duration, seed=i + 1, **vk):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        t = np.arange(int(args.duration * args.fps)) / args.fps
        np.savez(d / "truth.npz", beats=beats, pulse=synth.pulse_from_beats(t, beats), fps=args.fps, rhythm=rhythm)
        print(f"{name}: {len(beats)} beats, mean {60 / np.mean(np.diff(beats)):.0f} BPM")


if __name__ == "__main__":
    main()
