"""Command line.

    afscreen run VIDEO [--variant full] [--debug-video out.mp4]   one video: heart rate, confidence, beats
    afscreen extract DATASET ROOT [--out DIR] [--workers N]        cache patch traces for a dataset
    afscreen evaluate CACHE_DIR [--out DIR]                       score every variant against the reference
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _extract_one(dataset: str, root: str, index: int, out_dir: str, max_side: int | None) -> str:
    from afscreen.io.datasets import LOADERS, downscale
    from afscreen.traces import extract

    rec = LOADERS[dataset](root)[index]
    out = Path(out_dir) / f"{rec.id}.npz"
    if out.exists():
        return f"{rec.id}: cached"
    tr = extract(downscale(rec.frames(), max_side), rec.fps)
    tr.meta = {"dataset": rec.dataset, "id": rec.id, "subject": rec.subject, **{k: v for k, v in rec.meta.items()
                                                                               if k != "beats"}}
    tr.ref_pulse = rec.aligned_bvp(tr.n_frames)
    if "beats" in rec.meta:
        tr.ref_beats = np.asarray(rec.meta["beats"], dtype=np.float64)
    tr.save(out)
    return f"{rec.id}: {tr.n_frames} frames, face found in {tr.detection_rate:.0%}"


def cmd_extract(a) -> None:
    from afscreen.io.datasets import LOADERS

    recs = LOADERS[a.dataset](a.root)
    if not recs:
        raise SystemExit(f"no recordings found under {a.root} (see docs/DATA.md for the expected layout)")
    n = len(recs) if a.limit is None else min(a.limit, len(recs))
    out_dir = a.out or f"cache/traces/{a.dataset}"
    print(f"{n} recordings -> {out_dir}")
    if a.workers <= 1:
        for i in range(n):
            print(_extract_one(a.dataset, a.root, i, out_dir, a.max_side), flush=True)
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            for msg in ex.map(_extract_one, [a.dataset] * n, [a.root] * n, range(n), [out_dir] * n, [a.max_side] * n):
                print(msg, flush=True)


def cmd_evaluate(a) -> None:
    from afscreen.eval.benchmark import run
    from afscreen.pipeline import VARIANTS

    variants = tuple(VARIANTS) if a.variants == "all" else tuple(a.variants.split(","))
    out = a.out or f"results/phase1/{Path(a.cache_dir).name}"
    summary = run(a.cache_dir, out, variants, a.level)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nwritten to {out}/")


def cmd_run(a) -> None:
    import cv2

    from afscreen.face import patches
    from afscreen.io.datasets import downscale, iter_video, video_fps
    from afscreen.pipeline import analyse
    from afscreen.traces import extract

    fps = video_fps(a.video)
    frames, points = [], []

    def keep(i, frame, face, rgb):
        if a.debug_video:
            frames.append(frame)
            points.append(None if face is None else face.points)

    tr = extract(downscale(iter_video(a.video), a.max_side), fps, on_frame=keep)
    r = analyse(tr, a.variant)
    print(f"video: {a.video}  ({tr.n_frames} frames at {fps:.1f} fps, face found in {tr.detection_rate:.0%})")
    print(f"patches: {r.level}   heart rate: {r.hr_bpm:.1f} BPM (spectrum), {r.hr_beats_bpm:.1f} BPM (beats)   "
          f"beats: {len(r.beats)}   confidence: {r.mean_confidence:.2f}")
    if a.debug_video and frames:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(a.debug_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        win = r.fusion.weights
        for i, (frame, pts) in enumerate(zip(frames, points)):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if pts is not None:
                k = int(np.argmin(np.abs(r.window_t - i / fps)))
                bgr = patches.draw(bgr, pts, r.level, win[k])
            writer.write(bgr)
        writer.release()
        print(f"debug video: {a.debug_video} (patch colour: green = trusted, red = not)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="afscreen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyse one video")
    r.add_argument("video")
    r.add_argument("--variant", default="full")
    r.add_argument("--max-side", type=int, default=720, help="downscale larger frames to this size")
    r.add_argument("--debug-video", help="write a video with the patch trust map drawn on the face")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("extract", help="cache patch traces for a dataset")
    e.add_argument("dataset", choices=["ubfc", "pure", "synthetic"])
    e.add_argument("root")
    e.add_argument("--out")
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--limit", type=int)
    e.add_argument("--max-side", type=int, default=None)
    e.set_defaults(func=cmd_extract)

    v = sub.add_parser("evaluate", help="score every variant on cached traces")
    v.add_argument("cache_dir")
    v.add_argument("--out")
    v.add_argument("--variants", default="all", help="comma-separated, or 'all'")
    v.add_argument("--level", type=int, choices=[25, 12, 6], help="force a patch level (default: by face size)")
    v.set_defaults(func=cmd_evaluate)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
