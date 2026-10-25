"""MIMIC PERform datasets (Charlton et al., 2022), read straight from the CSV zips as downloaded
from Zenodo or from their unpacked folders. Fingertip PPG, ECG and (where recorded) impedance
respiration from bedside monitors, at 125 Hz.

    MIMIC PERform AF       mimic_perform_af_csv.zip, mimic_perform_non_af_csv.zip
                           20 min, 35 ICU patients: 19 in AF, 16 not          (Phase 3)
    MIMIC PERform Testing  mimic_perform_test_all_csv.zip
                           10 min, 200 patients: 100 adults, 100 neonates     (Phase 2: breathing)
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FS = 125.0


@dataclass
class MimicRecording:
    id: str  # e.g. af_005, non_af_013, test_all_056
    af: bool
    fs: float
    ppg: np.ndarray
    ecg: np.ndarray | None
    resp: np.ndarray | None = None  # impedance pneumography, if recorded
    group: str = ""  # Testing dataset: "a" adult, "n" neonate


def _parse(rid: str, af: bool, raw: bytes, group: str = "") -> MimicRecording:
    df = pd.read_csv(io.BytesIO(raw))
    t = df["Time"].to_numpy(dtype=np.float64)
    fs = float(round(1.0 / np.median(np.diff(t)))) if len(t) > 1 else FS
    col = lambda c: df[c].to_numpy(dtype=np.float64) if c in df else None  # noqa: E731
    return MimicRecording(id=rid, af=af, fs=fs, ppg=col("PPG"), ecg=col("ECG"), resp=col("resp"), group=group)


def _members(root: Path, zip_glob: str, csv_pat: re.Pattern):
    """(file name, bytes reader) for every matching CSV in zips and folders under ``root``."""
    for z in sorted(root.rglob(zip_glob)):
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            fixes = {n: zf.read(n) for n in names if n.endswith("_fix.txt")}
            for name in names:
                if csv_pat.search(name):
                    yield name, zf.read(name), fixes
    for f in sorted(root.rglob("mimic_perform_*_data.csv")):
        if csv_pat.search(f.name):
            fix = f.with_name(f.name.replace("_data.csv", "_fix.txt"))
            yield f.name, f.read_bytes(), {fix.name: fix.read_bytes()} if fix.exists() else {}


def load_mimic_af(root: str | Path) -> list[MimicRecording]:
    """Every AF and non-AF recording under ``root`` (zips or unpacked CSVs, searched recursively)."""
    pat = re.compile(r"mimic_perform_(non_af|af)_(\d+)_data\.csv$")
    found: dict[str, MimicRecording] = {}
    for name, raw, _ in _members(Path(root), "mimic_perform_*af_csv.zip", pat):
        m = pat.search(name)
        rid = f"{m.group(1)}_{m.group(2)}"
        if rid not in found:
            found[rid] = _parse(rid, m.group(1) == "af", raw)
    return [found[k] for k in sorted(found)]


def load_mimic_testing(root: str | Path, group: str | None = "a") -> list[MimicRecording]:
    """MIMIC PERform Testing recordings (``group`` "a" = adults, "n" = neonates, None = both)."""
    pat = re.compile(r"mimic_perform_test_all_(\d+)_data\.csv$")
    found: dict[str, MimicRecording] = {}
    for name, raw, fixes in _members(Path(root), "mimic_perform_test_all_csv.zip", pat):
        num = pat.search(name).group(1)
        fix = next((v for k, v in fixes.items() if k.endswith(f"test_all_{num}_fix.txt")), b"")
        g = re.search(rb"Group: *(\w*)", fix)
        g = g.group(1).decode() if g else ""
        rid = f"test_all_{num}"
        if (group is None or g == group) and rid not in found:
            found[rid] = _parse(rid, False, raw, g)
    return [found[k] for k in sorted(found)]
