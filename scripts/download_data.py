"""Download the public datasets that can be fetched without a request form.

    python scripts/download_data.py mitdb          # MIT-BIH Arrhythmia (PhysioNet, ~100 MB) -> data/MIT-BIH

Everything else (UBFC-rPPG, PURE, UBFC-Phys, OBF, COHFACE, MIMIC PERform AF) is listed in
docs/DATA.md with where to get it and the folder layout the loaders expect.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def mitdb(out: Path) -> None:
    import wfdb

    out.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(out))
    print(f"MIT-BIH Arrhythmia -> {out} ({len(list(out.glob('*.hea')))} records)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["mitdb"])
    ap.add_argument("--out", help="target folder (default: data/<name>)")
    a = ap.parse_args()
    if a.dataset == "mitdb":
        mitdb(Path(a.out or "data/MIT-BIH"))


if __name__ == "__main__":
    main()
