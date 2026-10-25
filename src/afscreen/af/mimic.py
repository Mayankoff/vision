"""MIMIC PERform AF (Charlton et al., 2022): 20-min fingertip PPG + ECG from 35 ICU patients,
19 in AF and 16 not, at 125 Hz. Read straight from the CSV zips as downloaded from Zenodo
(mimic_perform_af_csv.zip, mimic_perform_non_af_csv.zip) or from their unpacked folders.
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
    id: str  # e.g. af_005, non_af_013
    af: bool
    fs: float
    ppg: np.ndarray
    ecg: np.ndarray | None


def _parse(name: str, raw: bytes) -> MimicRecording:
    m = re.search(r"mimic_perform_(non_af|af)_(\d+)_data\.csv$", name)
    df = pd.read_csv(io.BytesIO(raw))
    t = df["Time"].to_numpy(dtype=np.float64)
    fs = float(round(1.0 / np.median(np.diff(t)))) if len(t) > 1 else FS
    return MimicRecording(id=f"{m.group(1)}_{m.group(2)}", af=m.group(1) == "af", fs=fs,
                          ppg=df["PPG"].to_numpy(dtype=np.float64),
                          ecg=df["ECG"].to_numpy(dtype=np.float64) if "ECG" in df else None)


def load_mimic_af(root: str | Path) -> list[MimicRecording]:
    """Every AF and non-AF recording under ``root`` (zips or unpacked CSVs, searched recursively)."""
    root = Path(root)
    pat = re.compile(r"mimic_perform_(non_af|af)_\d+_data\.csv$")
    found: dict[str, MimicRecording] = {}
    for z in sorted(root.rglob("mimic_perform_*af_csv.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if pat.search(name):
                    r = _parse(name, zf.read(name))
                    found.setdefault(r.id, r)
    for f in sorted(root.rglob("mimic_perform_*_data.csv")):
        if pat.search(f.name):
            r = _parse(f.name, f.read_bytes())
            found.setdefault(r.id, r)
    return [found[k] for k in sorted(found)]
