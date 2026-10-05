"""Report figure: the pipeline on one video, stage by stage.

    python scripts/figure_pipeline.py data/UBFC-rPPG/subject12/vid.avi --gt data/UBFC-rPPG/subject12/ground_truth.txt --out fig_pipeline.png
    python scripts/figure_pipeline.py my_face.mp4 --out fig_pipeline.png          # no ground truth

Panels: (a) a frame with the forehead/cheek ROIs, (b) the skin-ROI RGB traces,
(c) recovered pulse for two methods, (d) their spectra with the estimated (and,
if given, true) heart rate.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from rppg.eval import plotstyle
from rppg.face.landmarker import FaceLandmarker
from rppg.face.roi import draw_rois
from rppg.io.datasets import downscale, iter_video, video_fps
from rppg.pipeline import estimate, extract_traces, reference_hr
from rppg.signal.hr import power_spectrum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--gt", help="UBFC-style ground_truth.txt (first line = reference BVP)")
    ap.add_argument("--methods", default="green,pos", help="two methods to contrast")
    ap.add_argument("--seconds", type=float, default=10.0, help="length of the time-domain panels")
    ap.add_argument("--max-side", type=int, default=720, help="downscale larger frames to this size (0 = never)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fps = video_fps(args.video)
    frame_idx = None
    with FaceLandmarker() as lm:
        tr = extract_traces(downscale(iter_video(args.video), args.max_side), fps, lm)
        # Re-detect on a middle frame for panel (a).
        for i, frame in enumerate(downscale(iter_video(args.video), args.max_side)):
            if i == tr.n_frames // 2:
                frame_idx = i
                break
    with FaceLandmarker(mode="image") as lm_img:
        pts = lm_img.detect(frame)
    overlay = cv2.cvtColor(draw_rois(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), pts), cv2.COLOR_BGR2RGB) if pts is not None else frame
    if pts is not None:  # crop to the face
        (x0, y0), (x1, y1) = pts.min(0).astype(int), pts.max(0).astype(int)
        m = int(0.25 * (x1 - x0))
        overlay = overlay[max(0, y0 - m): y1 + m, max(0, x0 - m): x1 + m]

    rgb, fs = tr.signal("skin")
    methods = args.methods.split(",")
    results = {m: estimate(rgb, fs, m) for m in methods}
    hr_true = None
    if args.gt:
        with open(args.gt) as f:
            bvp = np.array(f.readline().split(), dtype=float)
        bvp = np.interp(np.linspace(0, 1, len(rgb)), np.linspace(0, 1, len(bvp)), bvp)
        _, hr_true, _ = reference_hr(bvp, fs)

    plotstyle.apply()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.6, 1.6])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_rgb, ax_pulse = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax_spec = fig.add_subplot(gs[:, 2])

    ax_img.imshow(overlay)
    ax_img.set_axis_off()
    ax_img.set_title(f"(a) ROIs, frame {frame_idx}", loc="left")

    n = min(len(rgb), int(args.seconds * fs))
    t = np.arange(n) / fs
    for c, (name, color) in enumerate(zip("RGB", ("#e34948", "#008300", "#2a78d6"))):
        x = rgb[:n, c]
        ax_rgb.plot(t, (x - x.mean()) / x.mean() * 100, lw=1.2, color=color, label=name)
    ax_rgb.set_xlabel("time (s)")
    ax_rgb.set_title("(b) Mean skin colour (% change from mean)", loc="left")
    ax_rgb.legend(ncols=3, loc="upper right")

    for i, m in enumerate(methods):
        ax_pulse.plot(t, results[m].bvp[:n] - 3.5 * i, lw=1.2, color=plotstyle.SERIES[i], label=m.upper())
    ax_pulse.set_yticks([])
    ax_pulse.set_xlabel("time (s)")
    ax_pulse.set_title("(c) Recovered pulse (offset for clarity)", loc="left")
    ax_pulse.legend(ncols=len(methods), loc="upper right")

    for i, m in enumerate(methods):
        f, p = power_spectrum(results[m].bvp, fs)
        band = (f >= 0.6) & (f <= 3.2)
        ax_spec.plot(f[band] * 60, p[band] / p[band].max() + 1.15 * (len(methods) - 1 - i), lw=1.5,
                     color=plotstyle.SERIES[i], label=f"{m.upper()}: {results[m].hr_fft:.1f} BPM")
    if hr_true is not None:
        ax_spec.axvline(hr_true, color=plotstyle.TEXT, lw=1, ls="--", label=f"true HR: {hr_true:.1f} BPM")
    ax_spec.set_yticks([])
    ax_spec.set_xlabel("heart rate (BPM)")
    ax_spec.set_title("(d) Pulse spectrum (normalised)", loc="left")
    ax_spec.legend(loc="upper right")
    ax_spec.grid(True, axis="x")

    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}: " + ", ".join(f"{m.upper()} {r.hr_fft:.1f} BPM" for m, r in results.items())
          + (f", true {hr_true:.1f} BPM" if hr_true is not None else ""))


if __name__ == "__main__":
    main()
