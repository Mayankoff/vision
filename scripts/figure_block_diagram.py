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
# Stream colours (categorical slots 1-3): edge = full colour, fill = light tint.
STREAM = {
    "rgb": ("#2a78d6", "#e3eefb"),  # visible-light (RGB) stream
    "thermal": ("#eb6834", "#fde9df"),  # thermal stream
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
    W, H = 23.2, 13.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    ax.text(0.4, 12.55, "Proposed architecture: thermal-guided, quality-aware multi-patch rPPG with stress estimation",
            fontsize=16, fontweight="bold", color=INK)
    ax.text(0.4, 12.12, "RGB gives the pulse; thermal decides which skin to trust, gives breathing and nose-tip temperature; "
            "outputs are vital signs plus a stress level with a confidence", fontsize=10.5, color=INK2)

    bw, bh = 2.8, 1.75
    # Lane 1 - RGB stream
    y1 = 9.2
    rgb_in = box(ax, 0.4, y1, 2.3, bh, "RGB video", "webcam / dataset\n30 fps")
    lm = box(ax, 3.1, y1, bw, bh, "1. Face landmarks\n+ head pose", "MediaPipe, 478 3D points,\nrotation + translation", "rgb")
    pt = box(ax, 6.3, y1, bw, bh, "2. Skin patches", "~25 patches on forehead\nand cheeks (face mesh)", "rgb")
    tr = box(ax, 9.5, y1, bw, bh, "3. Patch colour traces", "mean R, G, B of each\npatch, every frame", "rgb")
    ps = box(ax, 12.7, y1, bw, bh, "4. POS per patch", "25 candidate\npulse signals", "rgb")
    for a, b in [(rgb_in, lm), (lm, pt), (pt, tr), (tr, ps)]:
        arrow(ax, right(a), left(b))

    # Lane 3 - thermal stream
    y3 = 1.0
    th_in = box(ax, 0.4, y3, 2.3, bh, "Thermal video", "LWIR camera\n(e.g. 160x120, ~9 fps)")
    reg = box(ax, 3.1, y3, bw, bh, "T1. RGB-thermal\nregistration", "map face landmarks and\npatches onto thermal frame", "thermal")
    troi = box(ax, 6.3, y3, bw, bh, "T2. Thermal ROIs", "patch temperatures,\nnose tip, nostrils", "thermal")
    nose = box(ax, 9.5, y3, bw, bh, "T3. Nose-tip\ntemperature", "slow cooling under\nstress (sympathetic)", "thermal")
    brth = box(ax, 12.7, y3, bw, bh, "T4. Breathing signal", "nostril temperature\noscillation -> RR", "thermal")
    for a, b in [(th_in, reg), (reg, troi), (troi, nose)]:
        arrow(ax, right(a), left(b))
    arrow(ax, top(troi, 0.8), top(brth, 0.3), rad=-0.32)  # thermal ROIs also feed the breathing signal
    arrow(ax, bottom(lm, 0.45), top(reg, 0.45))  # landmarks are transferred to the thermal frame
    ax.text(bottom(lm, 0.45)[0] + 0.12, 7.3, "landmarks", fontsize=8.3, color=INK2, va="center")

    # Lane 2 - trust scores (dashed group)
    yq, hq, cw = 4.95, 1.95, 1.84
    xs = [6.15 + i * (cw + 0.14) for i in range(5)]
    q_orient = box(ax, xs[0], yq, cw, hq, "Face\norientation", "turned away\n-> low", "core", 9.2, 7.9)
    q_therm = box(ax, xs[1], yq, cw, hq, "Thermal\nskin check", "not warm skin\n(hair, glasses)\n-> low", "core", 9.2, 7.9)
    q_bright = box(ax, xs[2], yq, cw, hq, "Brightness", "shadow or\nshine -> low", "core", 9.2, 7.9)
    q_pulse = box(ax, xs[3], yq, cw, hq, "Pulse\nquality", "sharp spectral\npeak -> high", "core", 9.2, 7.9)
    q_agree = box(ax, xs[4], yq, cw, hq, "Agreement", "vote on peak\nfrequency", "core", 9.2, 7.9)
    gx0, gx1 = xs[0] - 0.22, xs[4] + cw + 0.22
    ax.add_patch(FancyBboxPatch((gx0, yq - 0.25), gx1 - gx0, hq + 0.75, boxstyle="round,pad=0,rounding_size=0.15",
                                lw=1.2, ls="--", edgecolor=STREAM["core"][0], facecolor="none", zorder=0))
    ax.text(gx0 + 0.15, yq + hq + 0.12, "5. Trust score per patch, every few seconds", fontsize=9.6,
            fontweight="bold", color=STREAM["core"][0], va="bottom")
    arrow(ax, bottom(pt, 0.35), top(q_orient))
    arrow(ax, bottom(tr, 0.45), top(q_bright))
    arrow(ax, bottom(ps, 0.3), top(q_pulse))
    arrow(ax, right(q_pulse), left(q_agree))
    arrow(ax, top(troi, 0.55), bottom(q_therm))

    # Right part - fusion, rates, HRV
    cx, cwid = 16.4, 3.0
    fu = box(ax, cx, y1, cwid, bh, "6. Weighted fusion", "patch pulses x trust\n+ confidence", "core")
    mc = box(ax, cx, 6.75, cwid, bh, "7. Motion removal", "subtract the part that\nfollows head pose", "core")
    hr = box(ax, cx, 4.3, cwid, bh, "8. Heart + breathing\nrate", "FFT peak, avoiding\nbreathing harmonics", "core")
    hv = box(ax, cx, 1.85, cwid, bh, "9. HRV", "beat detection ->\nRMSSD, SDNN, LF/HF", "core")
    arrow(ax, right(ps), left(fu))
    arrow(ax, (gx1, yq + hq * 0.85), left(fu, 0.25))
    arrow(ax, bottom(fu), top(mc))
    arrow(ax, bottom(mc), top(hr))
    arrow(ax, bottom(hr), top(hv))
    arrow(ax, right(brth), left(hr, 0.3))
    # head pose -> motion removal, routed above the RGB lane
    path_arrow(ax, [top(lm, 0.75), (top(lm, 0.75)[0], 11.35), (19.75, 11.35), (19.75, mc[1] + bh * 0.5),
                    right(mc)], color=STREAM["rgb"][0])
    ax.text(10.5, 11.45, "head pose (motion reference)", fontsize=8.5, color=STREAM["rgb"][0], ha="center", va="bottom")

    # Stress estimation and output
    sx, sw = 20.2, 2.7
    st = box(ax, sx, 1.0, sw, 2.6, "10. Stress\nestimation", "features: HR, HRV, RR,\nnose-tip trend; only\nconfident windows;\nclassifier", "core")
    out = box(ax, sx, 6.0, sw, 4.95, "Output", "Heart rate (BPM)\nBreathing rate\nHRV (RMSSD, SDNN)\n\nStress level:\nlow / moderate / high\n\nConfidence, or\n\"signal unreliable\"")
    arrow(ax, right(hv), left(st, 0.6))
    arrow(ax, right(hr, 0.3), left(st, 0.92), rad=-0.15)
    arrow(ax, right(hr, 0.75), left(out, 0.12))
    arrow(ax, top(st), bottom(out))
    path_arrow(ax, [bottom(nose), (bottom(nose)[0], 0.45), (sx + sw * 0.5, 0.45), bottom(st)], color=STREAM["thermal"][0])
    ax.text(15.0, 0.52, "nose-tip temperature trend", fontsize=8.5, color=STREAM["thermal"][0], ha="center", va="bottom")

    # Legend
    for i, (k, label) in enumerate([("rgb", "RGB (visible light) stream"), ("thermal", "Thermal stream"),
                                    ("core", "Trust, fusion and analysis"), (None, "Input / output")]):
        x = 0.45
        y = 7.6 - i * 0.48
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
