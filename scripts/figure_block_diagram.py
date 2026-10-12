"""Report figure: block diagram of the proposed architecture.

    python scripts/figure_block_diagram.py --out docs/figures

Writes block_diagram_proposed.{png,svg}. Only our architecture is drawn;
existing methods (POS, CHROM, PhysNet, ...) are comparison baselines and are
not part of the diagram.
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
# Stream colours (categorical slots 1 and 3): edge = full colour, fill = light tint.
STREAM = {
    "rgb": ("#2a78d6", "#e3eefb"),  # visible-light (RGB) stream
    "core": ("#1baf7a", "#dcf4eb"),  # trust, fusion and analysis
    None: ("#b9b8b2", "#f1f0ec"),  # inputs and outputs
}


def box(ax, x, y, w, h, title, body="", kind=None, title_size=10.0, body_size=8.4):
    edge, fill = STREAM[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK, zorder=3, linespacing=1.15)
    if body:
        title_lines = title.count("\n") + 1
        ax.text(x + w / 2, y + h - 0.32 - 0.3 * title_lines, body, ha="center", va="top", fontsize=body_size,
                color=INK2, zorder=3, linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, p, q, color=LINE, rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=2, shrinkB=2))


def path_arrow(ax, pts, color=LINE, lw=1.4):
    """Polyline whose last segment ends in an arrow head (for routed connections)."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=1, solid_joinstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def right(b, f=0.5):
    return (b[0] + b[2], b[1] + b[3] * f)


def left(b, f=0.5):
    return (b[0], b[1] + b[3] * f)


def top(b, f=0.5):
    return (b[0] + b[2] * f, b[1] + b[3])


def bottom(b, f=0.5):
    return (b[0] + b[2] * f, b[1])


def proposed_diagram(path: Path) -> None:
    W, H = 26.4, 11.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    ax.text(0.4, 10.55, "Proposed architecture: quality-aware multi-patch rPPG for atrial fibrillation screening",
            fontsize=16, fontweight="bold", color=INK)
    ax.text(0.4, 10.12, "One ordinary RGB camera; each face patch is trusted only when its signal is reliable; "
            "outputs are vital signs, an AF screening result (primary) and a stress level (secondary), with a confidence",
            fontsize=10.5, color=INK2)

    bw, bh = 2.8, 1.75
    # Row 1 - from video to per-patch pulses
    y1 = 7.4
    rgb_in = box(ax, 0.4, y1, 2.3, bh, "RGB video", "webcam / phone /\ndataset, 30 fps")
    lm = box(ax, 3.1, y1, bw, bh, "1. Face landmarks\n+ head pose", "MediaPipe, 478 3D points,\nrotation + translation", "rgb")
    pt = box(ax, 6.3, y1, bw, bh, "2. Skin patches", "~25 patches on forehead\nand cheeks (face mesh)", "rgb")
    tr = box(ax, 9.5, y1, bw, bh, "3. Patch colour traces", "mean R, G, B of each\npatch, every frame", "rgb")
    ps = box(ax, 12.7, y1, bw, bh, "4. POS per patch", "25 candidate\npulse signals", "rgb")
    for a, b in [(rgb_in, lm), (lm, pt), (pt, tr), (tr, ps)]:
        arrow(ax, right(a), left(b))

    # Row 2 - trust scores (dashed group)
    yq, hq, cw = 3.55, 1.95, 2.3
    xs = [6.4, 9.35, 12.3, 14.75]
    q_orient = box(ax, xs[0], yq, cw, hq, "Face\norientation", "turned away\n-> low", "core", 9.4, 8.0)
    q_bright = box(ax, xs[1], yq, cw, hq, "Brightness", "shadow or\nshine -> low", "core", 9.4, 8.0)
    q_pulse = box(ax, xs[2], yq, cw, hq, "Pulse\nquality", "sharp spectral\npeak -> high", "core", 9.4, 8.0)
    q_agree = box(ax, xs[3], yq, cw - 0.45, hq, "Agreement", "vote on peak\nfrequency", "core", 9.4, 8.0)
    gx0, gx1 = xs[0] - 0.22, xs[3] + cw - 0.45 + 0.2
    ax.add_patch(FancyBboxPatch((gx0, yq - 0.25), gx1 - gx0, hq + 0.75, boxstyle="round,pad=0,rounding_size=0.15",
                                lw=1.2, ls="--", edgecolor=STREAM["core"][0], facecolor="none", zorder=0))
    ax.text(gx0 + 0.15, yq + hq + 0.12, "5. Trust score per patch, every few seconds", fontsize=9.6,
            fontweight="bold", color=STREAM["core"][0], va="bottom")
    arrow(ax, bottom(pt, 0.45), top(q_orient))
    arrow(ax, bottom(tr, 0.45), top(q_bright))
    arrow(ax, bottom(ps, 0.45), top(q_pulse))
    arrow(ax, right(q_pulse), left(q_agree))

    # Right part - fusion, rates, HRV
    cx, cwid = 17.35, 2.75
    fu = box(ax, cx, y1, cwid, bh, "6. Weighted fusion", "patch pulses x trust\n+ confidence", "core")
    mc = box(ax, cx, 5.05, cwid, bh, "7. Motion removal", "subtract the part that\nfollows head pose", "core")
    hr = box(ax, cx, 2.7, cwid, bh, "8. Breathing + heart\nrate", "breathing from pulse\nmodulation; HR peak\navoids its harmonics", "core",
             body_size=8.0)
    hv = box(ax, cx, 0.35, cwid, bh, "9. HRV", "beat detection ->\nRMSSD, SDNN, LF/HF", "core")
    arrow(ax, right(ps), left(fu))
    arrow(ax, (gx1, yq + hq * 0.95), left(fu, 0.2))
    arrow(ax, bottom(fu), top(mc))
    arrow(ax, bottom(mc), top(hr))
    arrow(ax, bottom(hr), top(hv))
    # head pose -> motion removal, routed above row 1
    path_arrow(ax, [top(lm, 0.75), (top(lm, 0.75)[0], 9.55), (20.33, 9.55), (20.33, mc[1] + bh * 0.8),
                    right(mc, 0.8)], color=STREAM["rgb"][0])
    ax.text(10.5, 9.63, "head pose (motion reference)", fontsize=8.5, color=STREAM["rgb"][0], ha="center", va="bottom")

    # Downstream analyses and output
    sx, sw, sh = 20.75, 2.45, 2.3
    af = box(ax, sx, 0.35, sw, sh, "10. AF screening\n(primary)", "beat-interval irregularity\n(RMSSD, entropy,\nPoincare); confident\nwindows only", "core",
             body_size=8.0)
    st = box(ax, sx, 2.95, sw, sh, "11. Stress\n(secondary)", "HR, HRV,\nbreathing rate;\nclassifier", "core", body_size=8.0)
    out = box(ax, 23.6, 0.35, 2.4, 8.8, "Output", "Heart rate (BPM)\nBreathing rate\nHRV (RMSSD,\nSDNN)\n\nRhythm:\nregular, or\n\"possible AF -\nsee a doctor\"\n\nStress level:\nlow / moderate /\nhigh\n\nConfidence, or\n\"signal\nunreliable\"",
              body_size=8.2)
    arrow(ax, right(hv), left(af, 0.38))
    arrow(ax, right(hr), left(st, 0.38))
    arrow(ax, right(hv, 0.85), left(st, 0.12))
    arrow(ax, right(af), (out[0], af[1] + sh / 2))
    arrow(ax, right(st), (out[0], st[1] + sh / 2))
    path_arrow(ax, [right(hr, 0.85), (20.55, hr[1] + bh * 0.85), (20.55, 6.15), (out[0], 6.15)])
    ax.text(22.15, 6.22, "vital signs", fontsize=8.3, color=INK2, ha="center", va="bottom")

    # Legend
    for i, (k, label) in enumerate([("rgb", "Video and face geometry"), ("core", "Trust, fusion and analysis (ours)"),
                                    (None, "Input / output")]):
        x, y = 0.45, 2.3 - i * 0.5
        ax.add_patch(FancyBboxPatch((x, y), 0.36, 0.28, boxstyle="round,pad=0,rounding_size=0.05",
                                    edgecolor=STREAM[k][0], facecolor=STREAM[k][1], lw=1.6))
        ax.text(x + 0.5, y + 0.14, label, va="center", fontsize=9, color=INK2)

    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()
    plt.rcParams["axes.grid"] = False
    proposed_diagram(out / "block_diagram_proposed")
    print(f"written to {out}/: block_diagram_proposed.png/.svg")


if __name__ == "__main__":
    main()
