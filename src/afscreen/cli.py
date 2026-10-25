"""Command line.

    afscreen run VIDEO [--variant full] [--debug-video out.mp4]   one video: heart rate, confidence, beats
    afscreen extract DATASET ROOT [--out DIR] [--workers N]        cache patch traces for a dataset
    afscreen evaluate CACHE_DIR [--out DIR]                       score every variant against the reference
    afscreen af-train MIMIC_DIR [--out DIR]                       train / validate the AF classifier (Phase 3)
    afscreen af-screen CACHE_DIR ... [--out DIR]                  AF false-alarm rate on normal-rhythm face videos
    afscreen extra-beats MITBIH_DIR [--mimic DIR]                 extra-beat rule on MIT-BIH (Phase 4)
    afscreen hr-flags MIMIC_DIR                                   slow / fast flags vs ECG (Phase 4)
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


def cmd_af_train(a) -> None:
    from afscreen.af.experiment import run

    results = run(a.root, a.out, a.window, a.camera_fs or None, jitter_ms=a.jitter_ms)
    for label, res in results.items():
        print(f"\n== {label}")
        print(res.dropna(axis=1, how="all").to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nwritten to {a.out}/")


def cmd_af_screen(a) -> None:
    from afscreen.af.screen import false_alarm_summary, load_model, screen_videos

    files = sorted({f for d in a.cache_dirs for f in Path(d).glob("*.npz")})
    if not files:
        raise SystemExit("no cached traces found")
    ectopy = None
    if a.extra_beats and Path(a.extra_beats, "chosen_removal.txt").exists():
        from afscreen.flags.experiment import chosen_rule

        ectopy = chosen_rule(a.extra_beats)
        print(f"extra beats removed first: {ectopy}")
    windows, videos = screen_videos(files, load_model(a.model), min_quality=a.min_quality, ectopy=ectopy)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    windows.to_csv(out / "windows.csv", index=False)
    videos.to_csv(out / "videos.csv", index=False)
    summary = false_alarm_summary(windows, videos)
    summary.to_csv(out / "summary.csv", index=False)
    fmt = lambda x: f"{x:.2f}"  # noqa: E731
    print(videos.to_string(index=False, float_format=fmt))
    print("\n(all videos are normal rhythm: every 'possible AF' is a false alarm)")
    print(summary.to_string(index=False, float_format=fmt))
    print(f"\nwritten to {out}/")


def cmd_extra_beats(a) -> None:
    from afscreen.flags.experiment import chosen_rule, run_ectopy, run_mimic_removal

    r = run_ectopy(a.mitbih, a.out)
    print(f"rule: {r['rule']}\n\nDS2 test:\n{r['test'].to_string(index=False, float_format=lambda x: f'{x:.3f}')}")
    print(f"\nremoval chosen on DS1: {r['chosen']}")
    if a.mimic:
        res = run_mimic_removal(a.mimic, Path(a.out).parent / "af_with_removal", chosen_rule(a.out))
        print(res.dropna(axis=1, how="all").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwritten to {a.out}/")


def cmd_hr_flags(a) -> None:
    from afscreen.flags.experiment import run_hr_flags

    print(run_hr_flags(a.mimic, a.out).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwritten to {a.out}/")


def cmd_vitals(a) -> None:
    from afscreen.vitals import run_breathing, run_heart_rate, run_hrv, write_summary

    hrv = run_hrv(a.cache_dirs, a.out) if a.cache_dirs else None
    br = run_breathing(a.mimic, a.out) if a.mimic else None
    hr = run_heart_rate(a.mimic, a.out) if a.mimic else None
    write_summary(a.out, br, hr, hrv)
    for name, df in (("HRV", hrv), ("heart rate", hr)):
        if df is not None:
            print(f"\n== {name}\n{df.to_string(index=False, float_format=lambda x: f'{x:.2f}')}")
    print(f"\nwritten to {a.out}/ (tables in summary_tables.md)")


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
    e.add_argument("dataset", choices=["ubfc", "ubfc1", "pure", "synthetic"])
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

    t = sub.add_parser("af-train", help="train and validate the AF classifier on MIMIC PERform AF")
    t.add_argument("root", help="folder with the MIMIC PERform AF CSV zips")
    t.add_argument("--out", default="results/phase3/mimic")
    t.add_argument("--window", type=float, default=30.0, help="window length, s")
    t.add_argument("--camera-fs", type=float, default=30.0, help="also test at this frame rate (0 = skip)")
    t.add_argument("--jitter-ms", type=float, default=16.0,
                   help="camera-like beat-timing noise for the final model (0 = none)")
    t.set_defaults(func=cmd_af_train)

    sc = sub.add_parser("af-screen", help="AF screening of cached normal-rhythm face videos: false-alarm rate")
    sc.add_argument("cache_dirs", nargs="+")
    sc.add_argument("--model", default="results/phase3/mimic/af_classifier.joblib")
    sc.add_argument("--min-quality", type=float, default=0.8, help="beat-shape gate (chosen on MIMIC)")
    sc.add_argument("--extra-beats", default="results/phase4/mitbih",
                    help="folder of 'afscreen extra-beats' results whose rule removes extra beats first ('' = off)")
    sc.add_argument("--out", default="results/phase3/face")
    sc.set_defaults(func=cmd_af_screen)

    x = sub.add_parser("extra-beats", help="tune / test the extra-beat rule on MIT-BIH and its effect on AF (Phase 4)")
    x.add_argument("mitbih", help="folder with MIT-BIH (Kaggle zip / CSV or PhysioNet WFDB)")
    x.add_argument("--mimic", help="also re-test the AF classifier on MIMIC PERform AF with removal")
    x.add_argument("--out", default="results/phase4/mitbih")
    x.set_defaults(func=cmd_extra_beats)

    h = sub.add_parser("hr-flags", help="slow / fast heart-rate flags vs ECG on MIMIC PERform AF (Phase 4)")
    h.add_argument("mimic")
    h.add_argument("--out", default="results/phase4/hr_flags")
    h.set_defaults(func=cmd_hr_flags)

    vt = sub.add_parser("vitals", help="breathing, heart rate and HRV experiments (Phase 2)")
    vt.add_argument("--mimic", help="MIMIC PERform folder (Testing + AF zips) for breathing and heart rate")
    vt.add_argument("--cache-dirs", nargs="*", default=["cache/traces/ubfc_all"], help="face videos for HRV")
    vt.add_argument("--out", default="results/phase2")
    vt.set_defaults(func=cmd_vitals)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
