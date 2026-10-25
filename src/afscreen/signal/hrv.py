"""Step 9 - heart-rate variability from beat times: RMSSD (beat-to-beat change, the core AF
feature) and SDNN (overall spread, for information). LF/HF is not computed (see the plan)."""

from __future__ import annotations

import numpy as np

from afscreen.af.features import clean_ibis


def rmssd_ms(ibis: np.ndarray) -> float:
    x = clean_ibis(ibis)
    return float(np.sqrt(np.mean(np.diff(x) ** 2)) * 1000) if len(x) >= 3 else float("nan")


def sdnn_ms(ibis: np.ndarray) -> float:
    x = clean_ibis(ibis)
    return float(np.std(x, ddof=1) * 1000) if len(x) >= 3 else float("nan")
