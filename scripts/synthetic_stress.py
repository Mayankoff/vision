"""Synthetic stress tests of the trust checks and motion removal (patch traces with known beats).

Real UBFC videos have no shadowed, shiny, turned-away or moving faces, so the trust checks cannot
show their value there (Phase 5). These simulated scenarios do have such problems:

    clean regular 72 BPM; 8 noisy patches; 3 shadowed + 3 turned patches (all six also noisy);
    head motion in two colour directions; clean AF; AF with 8 noisy patches
    plus a motion sweep: frequency x artifact strength, with and without motion removal

    python scripts/synthetic_stress.py            -> results/phase5/synthetic_stress*.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from afscreen import synth
from afscreen.eval.metrics import beat_f1, ibi_error_ms
from afscreen.pipeline import VARIANTS, analyse
from afscreen.traces import PatchTraces

FS, DUR = 30.0, 40.0
NOISY = tuple(range(2, 10))
PROBLEM = tuple(range(2, 8))

SCENARIOS = {
    "Clean, regular 72 BPM": ("regular", {}),
    "8 noisy patches": ("regular", dict(noisy=NOISY)),
    "Shadowed and turned patches": ("regular", dict(noisy=PROBLEM, dark=(2, 3, 4), turned=(5, 6, 7))),
    "Head motion artifact": ("regular", dict(motion_hz=1.0, motion_artifact=0.009)),
    "AF, clean": ("af", {}),
    "AF, 8 noisy patches": ("af", dict(noisy=NOISY)),
}


def traces(rhythm: str, seed: int, **kw) -> PatchTraces:
    beats = synth.regular_beats(DUR, 72, seed=seed) if rhythm == "regular" else synth.af_beats(DUR, 90, seed=seed)
    d = synth.patch_traces(beats, FS, DUR, seed=seed + 100, **kw)
    n = d["rgb"].shape[0]
    return PatchTraces(fps=FS, rgb=d["rgb"], counts=np.full((n, 25), 500.0), facing=d["facing"], pose=d["pose"],
                       ref_beats=beats)


def score(tr: PatchTraces, variant: str) -> dict:
    r = analyse(tr, variant)
    true_hr = 60.0 / np.mean(np.diff(tr.ref_beats))
    return {"ibi_err_ms": ibi_error_ms(r.beats, tr.ref_beats), "beat_f1": beat_f1(r.beats, tr.ref_beats),
            "hr_err": abs(r.hr_beats_bpm - true_hr), "confidence": r.mean_confidence}


def main(out_dir: str = "results/phase5", seeds=(1, 2, 3)) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (rhythm, kw) in SCENARIOS.items():
        for seed in seeds:
            tr = traces(rhythm, seed, **kw)
            for v in VARIANTS:
                rows.append({"scenario": name, "seed": seed, "variant": v, **score(tr, v)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "synthetic_stress_runs.csv", index=False)
    tbl = df.pivot_table(index="scenario", columns="variant", values="ibi_err_ms", aggfunc="mean", sort=False)
    tbl = tbl[list(VARIANTS)]
    tbl.to_csv(out / "synthetic_stress_ibi.csv")
    print("Beat-interval error (ms), mean of", len(seeds), "seeds\n", tbl.round(1).to_string())
    both = df[df.variant.isin(["single", "full"])].groupby(["scenario", "variant"], sort=False)[
        ["hr_err", "beat_f1", "confidence"]].mean().unstack("variant")
    both.to_csv(out / "synthetic_stress_single_vs_full.csv")
    print(both.round(2).to_string())

    sweep = []
    for hz, art in ((0.8, 0.03), (1.0, 0.009), (1.0, 0.03), (1.3, 0.021)):
        for seed in seeds:
            tr = traces("regular", seed, motion_hz=hz, motion_artifact=art)
            a, b = score(tr, "+beats"), score(tr, "full")
            sweep.append({"motion_hz": hz, "artifact_x_pulse": round(art / 0.003), "seed": seed,
                          "ibi_without": a["ibi_err_ms"], "ibi_with": b["ibi_err_ms"],
                          "hr_without": a["hr_err"], "hr_with": b["hr_err"]})
    sw = pd.DataFrame(sweep).groupby(["motion_hz", "artifact_x_pulse"]).mean(numeric_only=True).drop(columns="seed")
    sw.to_csv(out / "synthetic_stress_motion.csv")
    print(sw.round(2).to_string())


if __name__ == "__main__":
    main()
