"""Block 9 - evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy import stats

from rppg.protocol import PROTOCOL
from rppg.signal.hr import power_spectrum


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


def snr_db(bvp: np.ndarray, fs: float, hr_gt_bpm: float, band=PROTOCOL.hr_band, tol_bpm: float = PROTOCOL.snr_tolerance_bpm) -> float:
    """Pulse SNR in dB (de Haan & Jeanne, 2013), as in rPPG-Toolbox.

    Ratio of spectral power within +/- ``tol_bpm`` of the ground-truth HR and
    its first harmonic, to the remaining power inside ``band``.
    """
    f, p = power_spectrum(bvp, fs)
    f0, tol = hr_gt_bpm / 60.0, tol_bpm / 60.0
    near_signal = (np.abs(f - f0) <= tol) | (np.abs(f - 2 * f0) <= tol)
    in_band = (f >= band[0]) & (f <= band[1])
    signal_power = p[near_signal].sum()
    noise_power = p[in_band & ~near_signal].sum()
    if noise_power == 0 or signal_power == 0:
        return float("nan")
    return float(10 * np.log10(signal_power / noise_power))


def summarize(pred, gt, snr=None) -> dict[str, float]:
    """All headline metrics for a set of per-video (or per-window) HR estimates."""
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(gt)
    pred, gt = pred[ok], gt[ok]
    abs_err = np.abs(pred - gt)
    out = {
        "n": int(ok.sum()),
        "MAE": mae(pred, gt),
        "MAE_se": float(abs_err.std(ddof=1) / np.sqrt(len(abs_err))) if len(abs_err) > 1 else float("nan"),
        "RMSE": rmse(pred, gt),
        "MAPE": mape(pred, gt),
        "Pearson": pearson(pred, gt),
    }
    if snr is not None:
        snr = np.asarray(snr, dtype=np.float64)
        out["SNR"] = float(np.nanmean(snr))
    return out
