"""Phase 0 - create the fixed, subject-independent train/val/test splits.

    python scripts/make_splits.py                       # write splits/*.json from the known subject lists
    python scripts/make_splits.py --check ubfc data/UBFC-rPPG   # verify a split against the data on disk

Splits are by subject (no subject appears in two partitions), ~70/15/15,
with a fixed seed. Commit the JSON files and never regenerate them once
experiments have started.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rppg.io.datasets import LOADERS

SEED = 2025
FRACTIONS = {"val": 0.15, "test": 0.15}

# Subject IDs as distributed. UBFC-rPPG DATASET_2 has 42 subjects with gaps in
# the numbering; PURE has 10 subjects x 6 sessions; MMPD has 33 subjects.
# Run --check once each dataset is downloaded to confirm these lists.
KNOWN_SUBJECTS = {
    "ubfc": [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27]
    + list(range(30, 50)),
    "pure": list(range(1, 11)),
    "mmpd": list(range(1, 34)),
}


def make_split(subjects: list[int], seed: int = SEED) -> dict[str, list[int]]:
    subjects = sorted(subjects)
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(FRACTIONS["test"] * n + 0.5))  # round half up
    n_val = max(1, int(FRACTIONS["val"] * n + 0.5))
    return {
        "train": sorted(shuffled[n_test + n_val :]),
        "val": sorted(shuffled[n_test : n_test + n_val]),
        "test": sorted(shuffled[:n_test]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="splits")
    ap.add_argument("--check", nargs=2, metavar=("DATASET", "ROOT"), help="verify splits/<DATASET>.json against data on disk")
    args = ap.parse_args()

    if args.check:
        name, root = args.check
        split = json.loads((Path(args.out) / f"{name}.json").read_text())
        on_disk = sorted({int(r.subject) for r in LOADERS[name](root)})
        in_split = sorted(s for part in ("train", "val", "test") for s in split[part])
        missing, extra = sorted(set(in_split) - set(on_disk)), sorted(set(on_disk) - set(in_split))
        print(f"{name}: {len(on_disk)} subjects on disk, {len(in_split)} in split")
        if missing or extra:
            raise SystemExit(f"MISMATCH - in split but not on disk: {missing}; on disk but not in split: {extra}")
        print("OK")
        return

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for name, subjects in KNOWN_SUBJECTS.items():
        split = make_split(subjects)
        doc = {"dataset": name, "seed": SEED, "unit": "subject", **split}
        (Path(args.out) / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
        print(f"{name}: train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])}")


if __name__ == "__main__":
    main()
