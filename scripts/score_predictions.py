"""Phase 2 (and 3) - score rPPG-Toolbox predictions with OUR frozen protocol.

    python scripts/score_predictions.py cache/toolbox/predictions/UBFC-rPPG --out results/toolbox/ubfc

Reads the per-video .npz files written by scripts/toolbox/dump_unsupervised.py
(toolbox environment) and writes the same CSV format as run_classical.py, so
toolbox and our results can be concatenated and compared directly:

  <out>_per_video.csv   one row per (video, method)
  <out>_summary.csv     MAE / RMSE / MAPE / Pearson / SNR per method
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rppg.eval.metrics import snr_db, summarize
from rppg.io.datasets import parse_toolbox_mmpd_name
from rppg.pipeline import reference_hr, score_raw_pulse

# Toolbox ROI = Haar-cascade face box enlarged 1.5x, resized to 72x72.
TOOLBOX_ROI = "toolbox_face_box"


def video_identity(dataset: str, filename: str) -> tuple[str, str, dict]:
    """Map a toolbox file name to (our video id, subject, labels)."""
    if dataset == "PURE":  # toolbox index 101 = our "01-01"
        n = int(filename)
        vid = f"{n // 100:02d}-{n % 100:02d}"
        return vid, vid.split("-")[0], {}
    if dataset == "MMPD":  # labels are encoded in the name; MMPD's video index is not
        subject, labels = parse_toolbox_mmpd_name(filename)
        return filename, subject, labels
    return filename, filename.replace("subject", ""), {}  # UBFC-rPPG: "subject7"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="directory of .npz files from dump_unsupervised.py")
    ap.add_argument("--out", required=True, help="output prefix, e.g. results/toolbox/ubfc")
    args = ap.parse_args()

    files = sorted(Path(args.predictions).glob("*.npz"))
    if not files:
        raise SystemExit(f"no prediction files in {args.predictions}")
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        fs, dataset, filename = float(z["fs"]), str(z["dataset"]), str(z["filename"])
        vid, subject, labels = video_identity(dataset, filename)
        _, gt_fft, gt_peak = reference_hr(z["gt_bvp"], fs)
        for key in (k for k in z.files if k.startswith("pred_")):
            method = key[5:]
            res = score_raw_pulse(z[key], fs)
            rows.append(
                {
                    "dataset": dataset,
                    "id": vid,
                    "subject": subject,
                    "impl": "toolbox",
                    "method": method.lower(),
                    "roi": TOOLBOX_ROI,
                    "hr_pred": res.hr_fft,
                    "hr_gt": gt_fft,
                    "hr_pred_peak": res.hr_peak,
                    "hr_gt_peak": gt_peak,
                    "snr_db": snr_db(res.bvp, fs, gt_fft),
                    "n_frames": len(z[key]),
                    "method_ms_per_frame": float(z[f"ms_per_frame_{method}"]),
                    **{f"label_{k}": v for k, v in labels.items()},
                }
            )
    df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [{"method": m, "roi": TOOLBOX_ROI, **summarize(g["hr_pred"], g["hr_gt"], g["snr_db"]),
          "method_ms_per_frame": g["method_ms_per_frame"].mean()} for m, g in df.groupby("method", sort=False)]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out}_per_video.csv", index=False)
    summary.to_csv(f"{out}_summary.csv", index=False)
    print(f"{len(files)} videos\n" + summary.round(2).to_string(index=False))
    print(f"\nwritten: {out}_per_video.csv, {out}_summary.csv")


if __name__ == "__main__":
    main()
