"""Phase 2 - comparison report: our classical pipeline vs rPPG-Toolbox, and the ROI ablation.

    python scripts/phase2_report.py results/classical/*_per_video.csv results/toolbox/*_per_video.csv --out results/phase2

Inputs are per-video CSVs from run_classical.py (ours) and score_predictions.py
(toolbox), for any mix of datasets. Writes to --out:

  master.csv              every (dataset, implementation, method, ROI) with all metrics
  cross_check.csv         ours vs toolbox, video by video, for the shared methods
  roi_ablation.csv        our methods on each ROI, with paired tests vs the skin ROI
  fig_mae_by_method.png   MAE: ours (skin), ours (face box), toolbox (face box)
  fig_roi_ablation.png    pulse SNR per ROI
  fig_ours_vs_toolbox.png per-video HR, ours vs toolbox
  REPORT.md               all tables in one place
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from rppg.eval import plotstyle
from rppg.eval.metrics import summarize

SHARED_METHODS = ["green", "ica", "chrom", "pos"]
ROI_ORDER = ["skin", "forehead", "left_cheek", "right_cheek", "face_box"]
VARIANTS = [("ours", "skin", "Ours · skin ROI"), ("ours", "face_box", "Ours · face box"),
            ("toolbox", "toolbox_face_box", "Toolbox · face box")]
LABEL_COLS = ["label_light", "label_motion", "label_exercise", "label_skin_color", "label_gender",
              "label_glasser", "label_hair_cover", "label_makeup"]


def match_key(df: pd.DataFrame) -> pd.Series:
    """Video identity shared by both codebases. MMPD videos are identified by subject +
    condition labels, because the toolbox drops MMPD's video index from its file names."""
    key = df["dataset"] + "/" + df["id"].astype(str)
    mmpd = df["dataset"] == "MMPD"
    if mmpd.any():
        cols = [c for c in LABEL_COLS if c in df]
        key[mmpd] = "MMPD/" + df.loc[mmpd, "subject"].astype(str) + "/" + df.loc[mmpd, cols].astype(str).agg("/".join, axis=1)
    return key


def paired_wilcoxon(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 6 or np.allclose(a, b):
        return float("nan")
    return float(stats.wilcoxon(a, b).pvalue)


def md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for col, v in zip(cols, row):
            if not isinstance(v, (float, np.floating)):
                cells.append(str(v))
            elif not np.isfinite(v):
                cells.append("–")
            elif "_p" in col:  # p-values
                cells.append("<0.001" if v < 0.001 else f"{v:.3f}")
            else:
                cells.append(format(v, floatfmt))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load(paths) -> pd.DataFrame:
    # Text dtype for ids and labels: otherwise e.g. skin type 5 reads as 5.0 in one file and 5 in another.
    text = {c: str for c in ["id", "subject", *LABEL_COLS]}
    df = pd.concat([pd.read_csv(p, dtype=text) for p in paths], ignore_index=True)
    df["impl"] = df["impl"].fillna("ours") if "impl" in df else "ours"
    df["subject"] = df["subject"].str.lstrip("0")
    df["key"] = match_key(df)
    return df


def master_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, impl, m, roi), g in df.groupby(["dataset", "impl", "method", "roi"], sort=False):
        rows.append({"dataset": ds, "impl": impl, "method": m, "roi": roi,
                     **summarize(g["hr_pred"], g["hr_gt"], g["snr_db"]),
                     "ms_per_frame": g["method_ms_per_frame"].mean()})
    return pd.DataFrame(rows)


def cross_check(df: pd.DataFrame) -> pd.DataFrame:
    tb = df[df["impl"] == "toolbox"]
    rows = []
    for roi in ("skin", "face_box"):
        ours = df[(df["impl"] == "ours") & (df["roi"] == roi)]
        j = ours.merge(tb, on=["key", "method"], suffixes=("_ours", "_tb"))
        for (ds, m), g in j.groupby(["dataset_ours", "method"], sort=False):
            diff = np.abs(g["hr_pred_ours"] - g["hr_pred_tb"])
            err_o, err_t = np.abs(g["hr_pred_ours"] - g["hr_gt_ours"]), np.abs(g["hr_pred_tb"] - g["hr_gt_tb"])
            rows.append({"dataset": ds, "method": m, "ours_roi": roi, "n": len(g),
                         "median_abs_hr_diff": float(np.median(diff)), "pct_within_3bpm": float(100 * np.mean(diff <= 3)),
                         "MAE_ours": float(err_o.mean()), "MAE_toolbox": float(err_t.mean()),
                         "wilcoxon_p": paired_wilcoxon(err_o, err_t)})
    return pd.DataFrame(rows)


def roi_ablation(df: pd.DataFrame) -> pd.DataFrame:
    ours = df[df["impl"] == "ours"]
    rows = []
    for (ds, m), g in ours.groupby(["dataset", "method"], sort=False):
        base = g[g["roi"] == "skin"].set_index("key")
        for roi in ROI_ORDER:
            r = g[g["roi"] == roi].set_index("key")
            if r.empty:
                continue
            common = base.index.intersection(r.index)
            err_r = np.abs(r.loc[common, "hr_pred"] - r.loc[common, "hr_gt"])
            err_b = np.abs(base.loc[common, "hr_pred"] - base.loc[common, "hr_gt"])
            rows.append({"dataset": ds, "method": m, "roi": roi, **summarize(r["hr_pred"], r["hr_gt"], r["snr_db"]),
                         "wilcoxon_p_vs_skin": paired_wilcoxon(err_r, err_b) if roi != "skin" else float("nan")})
    return pd.DataFrame(rows)


def fig_mae(master: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(master["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 4.0), squeeze=False, layout="constrained")
    for ax, ds in zip(axes[0], datasets):
        sub = master[master["dataset"] == ds]
        values = [[sub[(sub["impl"] == impl) & (sub["roi"] == roi) & (sub["method"] == m)]["MAE"].mean()
                   for m in SHARED_METHODS] for impl, roi, _ in VARIANTS]
        plotstyle.grouped_bars(ax, [m.upper() for m in SHARED_METHODS], [v[2] for v in VARIANTS], values)
        ax.set_title(f"{ds} (n = {int(sub['n'].max())} videos)", loc="left")
        ax.set_ylabel("MAE (BPM)")
    fig.suptitle("Heart-rate error by method and implementation (lower is better)", x=0.01, ha="left", fontsize=12)
    plotstyle.legend_below(fig, axes[0][0], ncols=len(VARIANTS))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_roi(abl: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(abl["dataset"]))
    rois = [r for r in ROI_ORDER if r in set(abl["roi"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.6 * len(datasets), 4.2), squeeze=False, layout="constrained")
    for ax, ds in zip(axes[0], datasets):
        sub = abl[abl["dataset"] == ds]
        values = [[sub[(sub["roi"] == r) & (sub["method"] == m)]["SNR"].mean() for m in SHARED_METHODS] for r in rois]
        plotstyle.grouped_bars(ax, [m.upper() for m in SHARED_METHODS], [r.replace("_", " ") for r in rois], values)
        ax.set_title(ds, loc="left")
        ax.set_ylabel("pulse SNR (dB)")
        ax.axhline(0, color=plotstyle.TEXT_SECONDARY, lw=0.8)
    fig.suptitle("Pulse signal quality by region of interest (higher is better)", x=0.01, ha="left", fontsize=12)
    plotstyle.legend_below(fig, axes[0][0], ncols=len(rois))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scatter(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    ours = df[(df["impl"] == "ours") & (df["roi"] == "face_box")]
    j = ours.merge(df[df["impl"] == "toolbox"], on=["key", "method"], suffixes=("_ours", "_tb"))
    if j.empty:
        return
    datasets = list(dict.fromkeys(j["dataset_ours"]))[:3]  # scatter: at most 3 colours stay distinguishable
    fig, axes = plt.subplots(1, len(SHARED_METHODS), figsize=(3.2 * len(SHARED_METHODS), 3.4), sharex=True, sharey=True,
                             layout="constrained")
    lo, hi = np.nanmin(j[["hr_pred_ours", "hr_pred_tb"]].values) - 5, np.nanmax(j[["hr_pred_ours", "hr_pred_tb"]].values) + 5
    for ax, m in zip(axes, SHARED_METHODS):
        ax.plot([lo, hi], [lo, hi], color=plotstyle.GRID, lw=1.5, zorder=0)
        for color, ds in zip(plotstyle.SERIES, datasets):
            g = j[(j["method"] == m) & (j["dataset_ours"] == ds)]
            ax.scatter(g["hr_pred_tb"], g["hr_pred_ours"], s=22, color=color, edgecolor=plotstyle.SURFACE, linewidth=1, label=ds)
        ax.set_title(m.upper(), loc="left")
        ax.set_xlabel("toolbox HR (BPM)")
        ax.grid(True, axis="both")
    axes[0].set_ylabel("our HR, face box (BPM)")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Per-video agreement: our implementation vs rPPG-Toolbox (same face-box ROI type)", x=0.01, ha="left", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="per-video CSVs from run_classical.py and score_predictions.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = load(args.csvs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()
    master, cc, abl = master_table(df), cross_check(df), roi_ablation(df)
    master.to_csv(out / "master.csv", index=False)
    cc.to_csv(out / "cross_check.csv", index=False)
    abl.to_csv(out / "roi_ablation.csv", index=False)
    fig_mae(master, out / "fig_mae_by_method.png")
    fig_roi(abl, out / "fig_roi_ablation.png")
    fig_scatter(df, out / "fig_ours_vs_toolbox.png")

    show = ["dataset", "impl", "method", "roi", "n", "MAE", "RMSE", "MAPE", "Pearson", "SNR", "ms_per_frame"]
    report = [
        "# Phase 2 results",
        "",
        f"Inputs: {', '.join(Path(p).name for p in args.csvs)}",
        "",
        "All numbers use the frozen protocol in `src/rppg/protocol.py` (whole-video FFT HR, 0.7–3.0 Hz).",
        "",
        "## 1. All methods",
        "",
        "![MAE by method](fig_mae_by_method.png)",
        "",
        md_table(master[show]),
        "",
        "## 2. Cross-check: ours vs rPPG-Toolbox (same videos)",
        "",
        "`median_abs_hr_diff` and `pct_within_3bpm` compare the two implementations' HR on each video. "
        "`wilcoxon_p` tests whether their absolute errors differ (paired; – when n < 6).",
        "",
        "![ours vs toolbox](fig_ours_vs_toolbox.png)",
        "",
        md_table(cc),
        "",
        "## 3. ROI ablation (our implementation)",
        "",
        "`wilcoxon_p_vs_skin` tests each ROI's absolute errors against the combined skin ROI on the same videos.",
        "",
        "![ROI ablation](fig_roi_ablation.png)",
        "",
        md_table(abl[["dataset", "method", "roi", "n", "MAE", "RMSE", "Pearson", "SNR", "wilcoxon_p_vs_skin"]]),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"written to {out}/: REPORT.md, master.csv, cross_check.csv, roi_ablation.csv, 3 figures")


if __name__ == "__main__":
    main()
