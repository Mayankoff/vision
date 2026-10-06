"""Report figures: block diagrams of the overall system and of the proposed method.

    python scripts/figure_block_diagram.py --out docs/figures

Writes block_diagram_system.{png,svg} and block_diagram_proposed.{png,svg}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from rppg.eval import plotstyle  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
LINE = "#8a8984"
NEUTRAL_FILL = "#f1f0ec"
NEUTRAL_EDGE = "#b9b8b2"
# Branch colours (categorical slots 1-3): edge = full colour, fill = light tint.
BRANCH = {
    "classical": ("#2a78d6", "#e3eefb"),
    "deep": ("#eb6834", "#fde9df"),
    "proposed": ("#1baf7a", "#dcf4eb"),
}


def box(ax, x, y, w, h, title, body="", kind=None, title_size=10.5):
    edge, fill = BRANCH[kind] if kind else (NEUTRAL_EDGE, NEUTRAL_FILL)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=title_size,
                fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.6,
                color=INK2, zorder=3, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=INK, zorder=3)
    return (x, y, w, h)


def arrow(ax, p, q, color=LINE, style="-|>", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=2, shrinkB=2))


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def new_canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    return fig, ax


def system_diagram(path: Path) -> None:
    fig, ax = new_canvas(17, 9.4)
    ax.text(0.4, 9.0, "Contactless heart-rate estimation: overall system", fontsize=15, fontweight="bold", color=INK)
    ax.text(0.4, 8.6, "Three approaches run on the same videos and are scored with one evaluation protocol",
            fontsize=10, color=INK2)

    # Input
    vid = box(ax, 0.4, 4.0, 2.1, 1.6, "Input video", "webcam, or dataset\nUBFC-rPPG / PURE / MMPD\n640x480, 30 fps")

    # Branch rows: y positions
    yc, yd, yp = 6.35, 4.05, 1.0
    h = 1.55

    # Classical
    c1 = box(ax, 3.2, yc, 2.4, h, "Face landmarks", "MediaPipe FaceLandmarker\n478 3D points", "classical")
    c2 = box(ax, 6.0, yc, 2.4, h, "Skin ROI", "forehead + both cheeks\npolygon masks", "classical")
    c3 = box(ax, 8.8, yc, 2.4, h, "RGB trace", "mean R, G, B per frame\ndetrend, normalise", "classical")
    c4 = box(ax, 11.6, yc, 2.4, h, "Classical rPPG", "GREEN, ICA,\nCHROM, POS", "classical")

    # Deep learning
    d1 = box(ax, 3.2, yd, 2.4, h, "Face crop", "Haar-cascade box x1.5\n(rPPG-Toolbox)", "deep")
    d2 = box(ax, 6.0, yd, 2.4, h, "Clip preparation", "resize 72x72 / 128x128\nnormalised frames", "deep")
    d3 = box(ax, 8.8, yd, 5.2, h, "Deep rPPG models", "PhysNet, TS-CAN, EfficientPhys, PhysFormer\n"
             "pretrained or trained on our subject splits", "deep")

    # Proposed
    p1 = box(ax, 3.2, yp, 2.4, h + 0.35, "Face landmarks\n+ head pose", "\n\nMediaPipe 3D points,\nrotation / translation",
             "proposed")
    p2 = box(ax, 6.0, yp, 2.4, h + 0.35, "~25 skin patches", "\n\neach patch: own RGB trace\n-> own POS pulse",
             "proposed")
    p3 = box(ax, 8.8, yp, 2.4, h + 0.35, "Quality weighting", "\n\nspectral peak sharpness,\nface orientation,\nbrightness",
             "proposed")
    p4 = box(ax, 11.6, yp, 2.4, h + 0.35, "Fusion + motion\ncompensation",
             "\n\nweighted sum of patches,\nremove head-motion part", "proposed")

    for a, b in [(c1, c2), (c2, c3), (c3, c4), (d1, d2), (d2, d3), (p1, p2), (p2, p3), (p3, p4)]:
        arrow(ax, right(a), left(b))
    for b, rad in [(c1, -0.15), (d1, 0.0), (p1, 0.15)]:
        arrow(ax, right(vid), left(b), rad=rad)

    # Common back end
    post = box(ax, 14.55, 3.3, 2.1, 3.1, "Pulse to\nheart rate",
               "\n\ndetrend\nband-pass\n0.7-3.0 Hz\n\nFFT peak\n-> BPM", None)
    arrow(ax, right(c4), (post[0], post[1] + 2.6), rad=-0.1)
    arrow(ax, right(d3), left(post))
    arrow(ax, right(p4), (post[0], post[1] + 0.5), rad=0.1)
    ax.text(15.6, 3.05, "+ confidence\n(proposed only)", ha="center", va="top", fontsize=8, color=BRANCH["proposed"][0])

    ev = box(ax, 14.55, 0.35, 2.1, 1.95, "Evaluation", "\n\nMAE, RMSE, MAPE,\nPearson r, SNR\nper MMPD condition\nms / frame", None)
    arrow(ax, bottom(post), (bottom(post)[0], top(ev)[1] + 0.85))

    # Legend
    for i, (k, label) in enumerate([("classical", "Classical (existing methods, own implementation)"),
                                    ("deep", "Deep learning (existing models, rPPG-Toolbox)"),
                                    ("proposed", "Proposed method (this project)")]):
        x = 0.4 + i * 4.9
        ax.add_patch(FancyBboxPatch((x, 0.1), 0.35, 0.28, boxstyle="round,pad=0,rounding_size=0.05",
                                    edgecolor=BRANCH[k][0], facecolor=BRANCH[k][1], lw=1.6))
        ax.text(x + 0.5, 0.24, label, va="center", fontsize=9, color=INK2)

    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor="white")
    plt.close(fig)


def proposed_diagram(path: Path) -> None:
    edge, fill = BRANCH["proposed"]
    fig, ax = new_canvas(17, 9.0)
    ax.text(0.4, 8.6, "Proposed method: quality-aware, motion-compensated multi-patch rPPG", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(0.4, 8.2, "Runs on a CPU, needs no training; POS is reused inside each patch",
            fontsize=10, color=INK2)

    # Row 1: from frame to per-patch pulses
    h, y = 1.7, 5.6
    v = box(ax, 0.4, y, 2.2, h, "Video frame", "\n\n30 fps RGB")
    lm = box(ax, 3.0, y, 2.6, h, "1. Face landmarks", "\n\nMediaPipe: 478 3D points\n+ head rotation / shift", "proposed")
    pt = box(ax, 6.0, y, 2.6, h, "2. Split into patches", "\n\n~25 patches on forehead\nand cheeks (from mesh)", "proposed")
    tr = box(ax, 9.0, y, 2.6, h, "3. Patch traces", "\n\nmean R, G, B of every\npatch, every frame", "proposed")
    ps = box(ax, 12.0, y, 2.6, h, "4. POS per patch", "\n\n25 candidate pulse\nsignals", "proposed")
    for p_, q_ in [(v, lm), (lm, pt), (pt, tr), (tr, ps)]:
        arrow(ax, right(p_), left(q_))

    # Row 2: the three checks, each directly under the block it uses
    yq, hq = 2.95, 1.75
    q1 = box(ax, 6.0, yq, 2.6, hq, "Face orientation", "\n\npatch normal from its 3D\npoints; turned away -> low",
             "proposed", title_size=10)
    q2 = box(ax, 9.0, yq, 2.6, hq, "Brightness check", "\n\ntoo dark (shadow) or\ntoo bright (shine) -> low",
             "proposed", title_size=10)
    q3 = box(ax, 12.0, yq, 2.6, hq, "Pulse quality", "\n\nFFT of each patch pulse:\nshare of power in the\nsharpest peak",
             "proposed", title_size=10)
    for src, dst in [(pt, q1), (tr, q2), (ps, q3)]:
        arrow(ax, bottom(src), top(dst))
    ax.add_patch(FancyBboxPatch((5.75, yq - 0.25), 9.1, hq + 0.5, boxstyle="round,pad=0,rounding_size=0.15",
                                lw=1.2, ls="--", edgecolor=edge, facecolor="none", zorder=0))
    ax.text(5.55, yq + hq / 2, "5. Trust score\nper patch\n(updated every\nfew seconds)", ha="right", va="center",
            fontsize=10, fontweight="bold", color=edge, linespacing=1.3)

    # Row 3: fusion, motion removal, heart rate
    yf, hf = 0.75, 1.55
    fu = box(ax, 6.0, yf, 2.6, hf, "6. Weighted fusion", "\n\npatch pulses (4)\nx trust scores (5)", "proposed")
    mc = box(ax, 9.0, yf, 2.6, hf, "7. Motion removal", "\n\nsubtract the part that\nfollows head movement", "proposed")
    hr = box(ax, 12.0, yf, 2.6, hf, "8. Heart rate", "\n\nband-pass, FFT peak\n-> BPM", None)
    out = box(ax, 15.0, yf, 1.7, hf, "Output", "\n\nBPM +\nconfidence", None)
    arrow(ax, (fu[0] + fu[2] / 2, yq - 0.25), top(fu))
    for p_, q_ in [(fu, mc), (mc, hr), (hr, out)]:
        arrow(ax, right(p_), left(q_))

    # Head pose -> motion removal, routed below row 3 so it crosses nothing
    x0, yb, x1 = lm[0] + 0.5, 0.3, mc[0] + mc[2] / 2
    ax.plot([x0, x0, x1], [lm[1], yb, yb], color=edge, lw=1.4, zorder=1, solid_joinstyle="round")
    arrow(ax, (x1, yb), (x1, yf), color=edge)
    ax.text(x0 + 0.15, 2.6, "head pose\n(motion reference)", fontsize=8.5, color=edge, ha="left", va="center")

    ax.text(out[0] + out[2] / 2, yf + hf + 0.15, "low trust ->\n\"signal unreliable\"", fontsize=8.5, color=INK2,
            ha="center", va="bottom")

    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()
    plt.rcParams["axes.grid"] = False
    system_diagram(out / "block_diagram_system")
    proposed_diagram(out / "block_diagram_proposed")
    print(f"written to {out}/: block_diagram_system.png/.svg, block_diagram_proposed.png/.svg")


if __name__ == "__main__":
    main()
