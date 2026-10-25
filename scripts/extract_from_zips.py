"""Cache patch traces for UBFC-rPPG videos straight from the Google Drive zip parts.

Drive splits a dataset folder into parts (DATASET_2-...-1-001.zip, -002, ...), each holding about
one video plus some other subjects' ground-truth files, so a partial download leaves videos and
their references in different parts. This script:

  1. reads every reference file (DATASET_2 ground_truth.txt, DATASET_1 gtdump.xmp) from all parts,
  2. unpacks one video at a time to a temporary folder, caches its traces, then deletes the video
     (each video is 1-2 GB uncompressed),
  3. writes  cache/traces/ubfc/   DATASET_2 videos with a reference
             cache/traces/ubfc1/  DATASET_1 videos with a reference
             cache/traces/noref/  videos without a reference (still usable for false-alarm
                                  rates, since every UBFC subject has a normal rhythm)

Loose videos (e.g. a part that unpacked to vid-002.avi) are added with --video PATH=SUBJECT,
e.g. --video "datasets/UBFC/dataset 1/vid-003.avi=5-gt".

    python scripts/extract_from_zips.py datasets/UBFC
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from afscreen.io.datasets import downscale, iter_video, read_ubfc1_truth, read_ubfc2_truth, video_fps
from afscreen.traces import extract

TRUTH_FILES = {"ground_truth.txt": "DATASET_2", "gtdump.xmp": "DATASET_1"}


def scan(root: Path) -> tuple[dict, dict]:
    """(references {(dataset, subject): bytes}, videos {(dataset, subject): (zip path, member)})."""
    refs, videos = {}, {}
    for z in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                parts = info.filename.split("/")
                if len(parts) != 3:
                    continue
                ds, subj, fname = parts
                if fname in TRUTH_FILES:
                    refs[(ds, subj)] = zf.read(info)
                elif fname.endswith(".avi"):
                    videos.setdefault((ds, subj), (z, info.filename))
    return refs, videos


def reference(ds: str, raw: bytes | None, n_frames: int) -> np.ndarray | None:
    """The reference pulse resampled to one sample per frame, or None."""
    if raw is None:
        return None
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write(raw)
    try:
        bvp, _ = (read_ubfc2_truth if ds == "DATASET_2" else read_ubfc1_truth)(f.name)
    finally:
        Path(f.name).unlink()
    if len(bvp) == n_frames:
        return bvp
    return np.interp(np.linspace(0, 1, n_frames), np.linspace(0, 1, len(bvp)), bvp)


def cache_one(video: Path, ds: str, subj: str, raw_ref: bytes | None, out_root: Path, max_side: int | None) -> str:
    folder = "noref" if raw_ref is None else ("ubfc" if ds == "DATASET_2" else "ubfc1")
    vid_id = subj if ds == "DATASET_2" else f"ubfc1_{subj}"
    out = out_root / folder / f"{vid_id}.npz"
    if out.exists():
        return f"{vid_id}: cached"
    fps = video_fps(str(video))
    tr = extract(downscale(iter_video(str(video)), max_side), fps)
    tr.meta = {"dataset": "UBFC-rPPG" if ds == "DATASET_2" else "UBFC-rPPG-1", "id": vid_id, "subject": vid_id}
    tr.ref_pulse = reference(ds, raw_ref, tr.n_frames)
    tr.save(out)
    return f"{vid_id} -> {folder}: {tr.n_frames} frames at {fps:.2f} fps, face found in {tr.detection_rate:.0%}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder holding the zip parts (searched recursively)")
    ap.add_argument("--out", default="cache/traces")
    ap.add_argument("--tmp", default=None, help="where to unpack videos (default: the system temp folder)")
    ap.add_argument("--video", action="append", default=[], help="loose video as PATH=SUBJECT (DATASET_1 subject)")
    ap.add_argument("--max-side", type=int, default=None)
    a = ap.parse_args(argv)

    refs, videos = scan(Path(a.root))
    print(f"{len(videos)} videos and {len(refs)} reference files in the zip parts")
    tmp = Path(tempfile.mkdtemp(prefix="afscreen_", dir=a.tmp))
    try:
        for (ds, subj), (zpath, member) in sorted(videos.items()):
            with zipfile.ZipFile(zpath) as zf:
                video = Path(zf.extract(member, tmp))
            try:
                print(cache_one(video, ds, subj, refs.get((ds, subj)), Path(a.out), a.max_side), flush=True)
            finally:
                video.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for spec in a.video:
        path, subj = spec.rsplit("=", 1)
        print(cache_one(Path(path), "DATASET_1", subj, refs.get(("DATASET_1", subj)), Path(a.out), a.max_side),
              flush=True)


if __name__ == "__main__":
    main()
