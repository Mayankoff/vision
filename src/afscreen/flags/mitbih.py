"""MIT-BIH Arrhythmia Database (Moody & Mark, 2001): 48 half-hour ECGs at 360 Hz with every beat
labelled. Only the beat annotations are used (beat times and types), so Phase 4 tests the
extra-beat rule on beat timing, the same information a camera gives.

Reads the PhysioNet WFDB files (``scripts/download_data.py mitdb``) or the Kaggle CSV version
(``<rec>annotations.txt``, also inside its zip).
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = 360.0

# Beat annotation symbols (everything else is a rhythm change, noise or comment).
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
# Premature (extra) beats: supraventricular (A, a, J, S) and ventricular (V) premature beats.
PREMATURE = set("AaJSV")
# Fusion beats are part premature, part normal: neither a hit nor a false alarm.
IGNORE = set("FQ?")
# Paced records are excluded, as in the AAMI standard.
PACED = {"102", "104", "107", "217"}
# Inter-patient split of de Chazal et al. (2004): tune on DS1, test on DS2.
DS1 = ("101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230").split()
DS2 = ("100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234").split()
AF_RHYTHMS = ("(AFIB", "(AFL")


@dataclass
class BeatRecord:
    id: str
    times: np.ndarray  # beat times (s)
    symbols: np.ndarray  # beat type per beat
    in_af: np.ndarray  # True where the beat lies in an AF / flutter episode


def _parse_txt(rid: str, text: str) -> BeatRecord:
    times, syms, in_af = [], [], []
    af = False
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        sample, sym = int(parts[1]), parts[2]
        aux = line.split("\t")[-1].strip() if "\t" in line else ""
        if sym == "+" and aux.startswith("("):
            af = aux.startswith(AF_RHYTHMS)
        elif sym in BEAT_SYMBOLS:
            times.append(sample / FS)
            syms.append(sym)
            in_af.append(af)
    return BeatRecord(rid, np.array(times), np.array(syms), np.array(in_af, dtype=bool))


def _from_wfdb(path: Path) -> BeatRecord:
    import wfdb

    ann = wfdb.rdann(str(path.with_suffix("")), "atr")
    times, syms, in_af = [], [], []
    af = False
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        if sym == "+" and aux.startswith("("):
            af = aux.rstrip("\x00").startswith(AF_RHYTHMS)
        elif sym in BEAT_SYMBOLS:
            times.append(s / ann.fs)
            syms.append(sym)
            in_af.append(af)
    return BeatRecord(path.stem, np.array(times), np.array(syms), np.array(in_af, dtype=bool))


def load_mitbih(root: str | Path, include_paced: bool = False) -> dict[str, BeatRecord]:
    """Beat annotations of every record under ``root`` (zip, Kaggle text files or WFDB)."""
    root = Path(root)
    recs: dict[str, BeatRecord] = {}
    pat = re.compile(r"(\d{3})annotations\.txt$")
    for z in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                m = pat.search(name)
                if m and m.group(1) not in recs:
                    recs[m.group(1)] = _parse_txt(m.group(1), io.TextIOWrapper(zf.open(name)).read())
    for f in sorted(root.rglob("*annotations.txt")):
        m = pat.search(f.name)
        if m and m.group(1) not in recs:
            recs[m.group(1)] = _parse_txt(m.group(1), f.read_text())
    for f in sorted(root.rglob("*.atr")):
        if f.stem not in recs:
            recs[f.stem] = _from_wfdb(f)
    if not include_paced:
        recs = {k: v for k, v in recs.items() if k not in PACED}
    return dict(sorted(recs.items()))
