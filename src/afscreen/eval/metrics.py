"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy import stats

from afscreen.config import SETTINGS
from afscreen.signal.spectrum import power_spectrum


def mae(pred, gt) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(gt))))


def rmse(pred, gt) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(gt)) ** 2)))


def mape(pred, gt) -> float:
    gt = np.asarray(gt, dtype=np.float64)
    return float(np.mean(np.abs((np.asarray(pred) - gt) / gt)) * 100.0)


def pearson(pred, gt) -> float:
    if len(pred) < 3 or np.std(pred) == 0 or np.std(gt) == 0:
        return float("nan")
    return float(stats.pearsonr(pred, gt)[0])


def snr_db(pulse: np.ndarray, fs: float, hr_gt_bpm: float, band=SETTINGS.hr_band, tol_bpm: float = 6.0) -> float:
    """Pulse SNR in dB (de Haan & Jeanne, 2013): power within +/- ``tol_bpm`` of the reference HR
    and its first harmonic, versus the rest of the band."""
    f, p = power_spectrum(pulse, fs)
    f0, tol = hr_gt_bpm / 60.0, tol_bpm / 60.0
    near = (np.abs(f - f0) <= tol) | (np.abs(f - 2 * f0) <= tol)
    inb = (f >= band[0]) & (f <= band[1])
    sig, noise = p[near].sum(), p[inb & ~near].sum()
    if noise == 0 or sig == 0:
        return float("nan")
    return float(10 * np.log10(sig / noise))


def summarize(pred, gt) -> dict[str, float]:
    """Headline heart-rate metrics over a set of estimates."""
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(gt)
    pred, gt = pred[ok], gt[ok]
    err = np.abs(pred - gt)
    return {
        "n": int(ok.sum()),
        "MAE": mae(pred, gt) if len(pred) else float("nan"),
        "MAE_se": float(err.std(ddof=1) / np.sqrt(len(err))) if len(err) > 1 else float("nan"),
        "RMSE": rmse(pred, gt) if len(pred) else float("nan"),
        "MAPE": mape(pred, gt) if len(pred) else float("nan"),
        "Pearson": pearson(pred, gt),
    }


def ibi_error_ms(pred_beats: np.ndarray, ref_beats: np.ndarray, tol_s: float = 0.15) -> float:
    """Mean absolute error (ms) of inter-beat intervals over beats matched to the reference."""
    pred_beats, ref_beats = np.asarray(pred_beats), np.asarray(ref_beats)
    if len(pred_beats) < 2 or len(ref_beats) < 2:
        return float("nan")
    idx = np.searchsorted(pred_beats, ref_beats)
    lo = np.clip(idx - 1, 0, len(pred_beats) - 1)
    hi = np.clip(idx, 0, len(pred_beats) - 1)
    near = np.where(np.abs(pred_beats[lo] - ref_beats) <= np.abs(pred_beats[hi] - ref_beats), lo, hi)
    ok = np.abs(pred_beats[near] - ref_beats) <= tol_s
    errs = [abs((pred_beats[near[i + 1]] - pred_beats[near[i]]) - (ref_beats[i + 1] - ref_beats[i]))
            for i in range(len(ref_beats) - 1) if ok[i] and ok[i + 1] and near[i + 1] == near[i] + 1]
    return float(np.mean(errs) * 1000) if errs else float("nan")


def beat_f1(pred_beats: np.ndarray, ref_beats: np.ndarray, tol_s: float = 0.15) -> float:
    """F1 of beat detection: missed and spurious beats both lower it (IBI error ignores them)."""
    from afscreen.signal.beats import match_count

    total = len(pred_beats) + len(ref_beats)
    return 2.0 * match_count(np.asarray(pred_beats), np.asarray(ref_beats), tol_s) / total if total else float("nan")
