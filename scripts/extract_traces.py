"""Phase 1 - run face landmarks + ROI averaging over a dataset and cache the RGB traces.

    python scripts/extract_traces.py ubfc data/UBFC-rPPG --out cache/traces/ubfc --workers 4

This is the slow step (MediaPipe on every frame). It writes one .npz per video
containing every ROI's RGB trace, the frame-aligned reference BVP and the
dataset's labels, and skips videos that are already done, so it can be
interrupted and resumed. Everything downstream (run_classical.py) reads only
these files.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rppg.face.landmarker import FaceLandmarker, ensure_model
from rppg.io.datasets import LOADERS
from rppg.pipeline import extract_traces


def process(dataset: str, root: str, index: int, out_dir: str, model: str, debug_dir: str | None) -> str:
    rec = LOADERS[dataset](root)[index]
    out = Path(out_dir) / f"{rec.id}.npz"
    if out.exists():
        return f"skip {rec.id}"
    t0 = time.perf_counter()
    debug = str(Path(debug_dir) / f"{rec.id}.mp4") if debug_dir else None
    with FaceLandmarker(model) as lm:
        traces = extract_traces(rec.frames(), rec.fps, lm, debug_video=debug)
    traces.meta = {
        "dataset": rec.dataset,
        "id": rec.id,
        "subject": rec.subject,
        "gt_bvp": rec.aligned_bvp(traces.n_frames),
        "extract_ms_per_frame": 1000 * (time.perf_counter() - t0) / traces.n_frames,
        **{f"label_{k}": v for k, v in rec.meta.items()},
    }
    tmp = out.with_name(out.stem + ".tmp.npz")  # atomic write: never leave a half-written cache file
    traces.save(tmp)
    tmp.rename(out)
    return f"done {rec.id}: {traces.n_frames} frames, face in {traces.detection_rate:.1%}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(LOADERS))
    ap.add_argument("root", help="raw dataset directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help="only the first N videos (smoke test)")
    ap.add_argument("--debug-videos", help="also write ROI-overlay videos to this directory")
    ap.add_argument("--model", help="face_landmarker.task path")
    args = ap.parse_args()

    recs = LOADERS[args.dataset](args.root)
    if not recs:
        raise SystemExit(f"no recordings found under {args.root}")
    n = min(len(recs), args.limit or len(recs))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.debug_videos:
        Path(args.debug_videos).mkdir(parents=True, exist_ok=True)
    model = str(ensure_model(args.model))
    print(f"{args.dataset}: {n} of {len(recs)} recordings -> {args.out}")

    jobs = [(args.dataset, args.root, i, args.out, model, args.debug_videos) for i in range(n)]
    if args.workers <= 1:
        for job in jobs:
            print(process(*job), flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            for fut in as_completed([pool.submit(process, *job) for job in jobs]):
                print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
