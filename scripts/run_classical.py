"""Phase 1 - run classical rPPG methods on cached traces and score them.

    python scripts/run_classical.py cache/traces/ubfc --out results/classical/ubfc
    python scripts/run_classical.py cache/traces/ubfc --split splits/ubfc.json --subset test --out results/classical/ubfc_test

Writes:
  <out>_per_video.csv   one row per (video, method, ROI): predicted/GT HR, SNR, runtime, labels
  <out>_summary.csv     MAE / RMSE / MAPE / Pearson / SNR per (method, ROI)
  <out>_windows.csv     per-window HR (only with --window)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rppg.eval.metrics import snr_db, summarize
from rppg.pipeline import Traces, estimate, reference_hr, windowed_pair
from rppg.signal.methods import METHODS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", help="directory of .npz files from extract_traces.py")
    ap.add_argument("--out", required=True, help="output prefix, e.g. results/classical/ubfc")
    ap.add_argument("--methods", default=",".join(METHODS), help=f"comma-separated, from {list(METHODS)}")
    ap.add_argument("--rois", default="skin,face_box")
    ap.add_argument("--split", help="splits/<dataset>.json; with --subset, evaluate only those subjects")
    ap.add_argument("--subset", choices=["train", "val", "test"])
    ap.add_argument("--window", type=float, help="also score HR per window of this many seconds")
    args = ap.parse_args()

    files = sorted(Path(args.traces).glob("*.npz"))
    if args.split and args.subset:
        keep = {str(s) for s in json.loads(Path(args.split).read_text())[args.subset]}
        files = [f for f in files if str(int(Traces.load(f).meta["subject"])) in keep]
    if not files:
        raise SystemExit("no trace files to evaluate")

    methods, rois = args.methods.split(","), args.rois.split(",")
    rows, win_rows = [], []
    for f in files:
        tr = Traces.load(f)
        meta = tr.meta
        gt_clean, gt_fft, gt_peak = reference_hr(np.asarray(meta["gt_bvp"]), tr.fps)
        labels = {k: v for k, v in meta.items() if k.startswith("label_")}
        for roi in rois:
            rgb, fs = tr.signal(roi)
            for m in methods:
                t0 = time.perf_counter()
                res = estimate(rgb, fs, m)
                runtime = time.perf_counter() - t0
                rows.append(
                    {
                        "dataset": meta["dataset"],
                        "id": meta["id"],
                        "subject": meta["subject"],
                        "method": m,
                        "roi": roi,
                        "hr_pred": res.hr_fft,
                        "hr_gt": gt_fft,
                        "hr_pred_peak": res.hr_peak,
                        "hr_gt_peak": gt_peak,
                        "snr_db": snr_db(res.bvp, fs, gt_fft),
                        "n_frames": tr.n_frames,
                        "face_detection_rate": tr.detection_rate,
                        "method_ms_per_frame": 1000 * runtime / tr.n_frames,
                        "extract_ms_per_frame": meta.get("extract_ms_per_frame", np.nan),
                        **labels,
                    }
                )
                if args.window:
                    hp, hg = windowed_pair(res.bvp, gt_clean[: len(res.bvp)], fs, args.window)
                    win_rows += [
                        {"id": meta["id"], "subject": meta["subject"], "method": m, "roi": roi, "window": i, "hr_pred": p, "hr_gt": g}
                        for i, (p, g) in enumerate(zip(hp, hg))
                    ]

    df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "method": m,
                "roi": roi,
                **summarize(g["hr_pred"], g["hr_gt"], g["snr_db"]),
                "MAE_peak": float(np.nanmean(np.abs(g["hr_pred_peak"] - g["hr_gt_peak"]))),
                "method_ms_per_frame": g["method_ms_per_frame"].mean(),
            }
            for (m, roi), g in df.groupby(["method", "roi"], sort=False)
        ]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out}_per_video.csv", index=False)
    summary.to_csv(f"{out}_summary.csv", index=False)
    if win_rows:
        wdf = pd.DataFrame(win_rows)
        wdf.to_csv(f"{out}_windows.csv", index=False)
        wsum = pd.DataFrame(
            [{"method": m, "roi": roi, **summarize(g["hr_pred"], g["hr_gt"])} for (m, roi), g in wdf.groupby(["method", "roi"], sort=False)]
        )
        wsum.to_csv(f"{out}_windows_summary.csv", index=False)
        print(f"\nPer-{args.window:g}s-window results:\n" + wsum.round(2).to_string(index=False))
    print(f"\n{len(files)} videos\n" + summary.round(2).to_string(index=False))
    print(f"\nwritten: {out}_per_video.csv, {out}_summary.csv")


if __name__ == "__main__":
    main()
