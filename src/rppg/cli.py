"""Offline prototype: facial video file -> heart rate (BPM).

    rppg video.avi                      # POS on forehead + cheeks
    rppg video.avi --method chrom --plot pulse.png --debug-video rois.mp4
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from rppg.face.landmarker import FaceLandmarker
from rppg.io.datasets import downscale, iter_video, video_fps
from rppg.pipeline import TRACE_NAMES, estimate, extract_traces
from rppg.signal.hr import power_spectrum, windowed_hr
from rppg.signal.methods import METHODS


def plot_result(path: str, result, window_s: float = 10.0, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(result.bvp)) / result.fs
    f, p = power_spectrum(result.bvp, result.fs)
    band = (f >= 0.5) & (f <= 4.0)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(t, result.bvp, lw=0.8)
    axes[0].set(xlabel="time (s)", ylabel="pulse (a.u.)", title=title or "Recovered pulse")
    axes[1].plot(f[band] * 60, p[band])
    axes[1].axvline(result.hr_fft, color="C3", ls="--", label=f"HR = {result.hr_fft:.1f} BPM")
    axes[1].set(xlabel="BPM", ylabel="power")
    axes[1].legend()
    wt, whr = windowed_hr(result.bvp, result.fs, window_s)
    axes[2].plot(wt, whr, marker=".")
    axes[2].set(xlabel="time (s)", ylabel="BPM", title=f"HR per {window_s:.0f} s window")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--method", default="pos", choices=sorted(METHODS))
    ap.add_argument("--roi", default="skin", choices=TRACE_NAMES)
    ap.add_argument("--fps", type=float, help="override the frame rate stored in the file")
    ap.add_argument("--plot", help="save waveform / spectrum / HR-over-time figure here")
    ap.add_argument("--debug-video", help="save a video with ROI overlays here")
    ap.add_argument("--model", help="path to face_landmarker.task (downloaded if missing)")
    ap.add_argument("--max-side", type=int, default=720, help="downscale larger frames to this size (0 = never)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.video):
        raise SystemExit(f"video not found: {os.path.abspath(args.video)}")
    fps = args.fps or video_fps(args.video)
    t0 = time.perf_counter()
    with FaceLandmarker(args.model) as lm:
        traces = extract_traces(downscale(iter_video(args.video), args.max_side), fps, lm, debug_video=args.debug_video)
    t1 = time.perf_counter()
    rgb, fs = traces.signal(args.roi)
    result = estimate(rgb, fs, args.method)
    t2 = time.perf_counter()

    print(f"frames: {traces.n_frames} @ {fs:.2f} fps, face detected in {traces.detection_rate:.1%} of frames")
    print(f"heart rate ({args.method.upper()}, {args.roi}): {result.hr_fft:.1f} BPM (FFT), {result.hr_peak:.1f} BPM (peaks)")
    print(f"time: {1000 * (t1 - t0) / traces.n_frames:.2f} ms/frame face+ROI, {1000 * (t2 - t1):.1f} ms total for {args.method}")
    if args.plot:
        plot_result(args.plot, result, title=f"{args.video} - {args.method.upper()}")
        print(f"plot saved to {args.plot}")


if __name__ == "__main__":
    main()
