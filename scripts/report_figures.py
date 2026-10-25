"""Figures for the final report, from the saved results (grayscale, like the DA3 report).

    python scripts/report_figures.py            -> docs/figures/final/*.png

The live-demo figure needs the UBFC video datasets/UBFC/dataset 1/vid-003.avi; the face is blurred
so that no dataset participant can be recognised in the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("docs/figures/final")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
EDGE = dict(edgecolor="black", linewidth=0.7)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_stress():
    t = pd.read_csv("results/phase5/synthetic_stress_ibi.csv", index_col=0)
    cols = [("single", "white", "", "Single region (baseline)"), ("patch_mean", "white", "///", "Patches, equal weights"),
            ("full", "0.25", "", "Proposed method (full)")]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, (c, fc, h, lab) in enumerate(cols):
        ax.bar(x + (i - 1) * 0.27, t[c], 0.27, color=fc, hatch=h, label=lab, **EDGE)
    ax.set_xticks(x, [s.replace(", ", ",\n").replace(" and ", " and\n") for s in t.index], fontsize=8.5)
    ax.set_ylabel("Beat-interval error (ms)")
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "fig_stress.png")


def fig_baselines():
    a = pd.read_csv("results/phase5/ablation.csv").set_index("variant")
    rows = [("single_green", "GREEN\n(one region)"), ("single_chrom", "CHROM\n(one region)"),
            ("single", "POS\n(one region)"), ("patch_mean", "POS + patches\n(equal weights)"),
            ("full", "Proposed\n(full)")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0))
    x = np.arange(len(rows))
    ax1.bar(x, [a.loc[r, "IBI_err_ms"] for r, _ in rows], 0.6, color=["white", "white", "0.6", "0.6", "0.25"], **EDGE)
    ax1.set_xticks(x, [lab for _, lab in rows], fontsize=7.5)
    ax1.set_ylabel("Beat-interval error (ms)")
    ax1.set_title("Beat timing (7 videos)", fontsize=10)
    vals = [int(a.loc[r, "AF_false_videos"]) for r, _ in rows]
    ax2.bar(x, vals, 0.6, color=["white", "white", "0.6", "0.6", "0.25"], **EDGE)
    for xi, v in zip(x, vals):
        ax2.text(xi, v + 0.15, str(v), ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, 12)
    ax2.set_xticks(x, [lab for _, lab in rows], fontsize=7.5)
    ax2.set_ylabel("Videos wrongly flagged as AF (of 12)")
    ax2.set_title("AF false alarms (12 healthy videos)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_baselines.png")


def fig_gating():
    g = pd.read_csv("results/phase3/mimic/gating_camera_30Hz_jitter16ms.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for gate, ls, lab in (("beat_shape", "-", "beat-shape consistency (used)"), ("peak_share", "--",
                                                                                  "spectral peak sharpness")):
        d = g[g.gate == gate]
        ax.plot(d.threshold, d.coverage_af, ls, color="black", marker="o", label=f"AF windows, {lab}")
        ax.plot(d.threshold, d.coverage_non_af, ls, color="0.55", marker="s", label=f"non-AF windows, {lab}")
    ax.set_xlabel("Quality threshold")
    ax.set_ylabel("Share of windows kept")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "fig_gating.png")


def fig_removal():
    d = pd.read_csv("results/phase4/mitbih/af_removal_DS2.csv", index_col=0)
    chosen = Path("results/phase4/mitbih/chosen_removal.txt").read_text().strip()
    cols = [("none", "No removal"), ("cap0.2", "Plain removal"), (chosen, "Removal of isolated\nextra beats (chosen)")]
    groups = [("AF", "AF windows detected\n(sensitivity)"), ("non-AF with extra beats",
                                                          "Normal rhythm with extra\nbeats called AF"),
              ("non-AF without", "Normal rhythm without extra\nbeats called AF")]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(groups))
    for i, ((c, lab), fc) in enumerate(zip(cols, ["white", "0.6", "0.25"])):
        ax.bar(x + (i - 1) * 0.27, [d.loc[g, c] for g, _ in groups], 0.27, color=fc, label=lab, **EDGE)
    ax.set_xticks(x, [lab for _, lab in groups], fontsize=8.5)
    ax.set_ylabel("Share of 30 s windows")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig_removal.png")


def fig_rmssd():
    d = pd.read_csv("results/phase2/hrv_per_video.csv")
    d = d[(d.variant == "single") & (d.refine == "parabolic")]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(d.rmssd_ref, d.rmssd, color="black", s=22)
    m = max(d.rmssd.max(), d.rmssd_ref.max()) * 1.1
    ax.plot([0, m], [0, m], "--", color="0.5", linewidth=0.8)
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_xlabel("RMSSD from the contact sensor (ms)")
    ax.set_ylabel("RMSSD from the camera (ms)")
    save(fig, "fig_rmssd.png")


def fig_demo(video="datasets/UBFC/dataset 1/vid-003.avi", at_s=45.0):
    import cv2

    from afscreen.face import patches
    from afscreen.io.datasets import iter_video, video_fps
    from afscreen.live import LiveMonitor, draw_status

    if not Path(video).exists():
        print("skipping the demo figure: video not found")
        return
    fps = video_fps(video)
    mon = LiveMonitor()
    st, face, frame = None, None, None
    for i, rgb in enumerate(iter_video(video)):
        t = i / fps
        face = mon.add_frame(rgb, t)
        if t >= at_s:
            frame = rgb
            st = mon.update()
            break
    mon.close()
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr = cv2.GaussianBlur(bgr, (0, 0), 12)  # no recognisable face in the report
    bgr = patches.draw(bgr, face.points, st.level, st.weights)
    cv2.imwrite(str(OUT / "fig_demo.png"), draw_status(bgr, st))


if __name__ == "__main__":
    fig_stress()
    fig_baselines()
    fig_gating()
    fig_removal()
    fig_rmssd()
    fig_demo()
    print("figures in", OUT)
