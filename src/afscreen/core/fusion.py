"""Steps 4-7 - POS per patch, trust-weighted fusion, motion removal and confidence.

    patch colours (T, P, 3) --POS--> P patch pulses
    every window (8 s, every 1 s), per patch:
        trust = facing x brightness x consistency^2 x beat agreement
    fused pulse = sum_k trust_k * pulse_k
    motion removal (before POS, on every patch's colour, and again on the fused pulse):
        subtract the part that a ridge regression on head pose explains
    confidence = geometric mean of patch consistency, beat-shape consistency, low motion, enough patches
    windows are overlap-added with a Hann window

Each ingredient can be switched off (FusionConfig) for the ablation study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from afscreen.config import SETTINGS, Settings
from afscreen.core.trust import beat_agreement_scores, brightness_weight, consistency, facing_weight, luma
from afscreen.signal.filters import bandpass, detrend, fill_gaps
from afscreen.signal.pos import pos
from afscreen.signal.quality import beat_shape_consistency

CONFIDENCE_PARTS = ("consistency", "beat_shape", "motion", "patches")


@dataclass(frozen=True)
class FusionConfig:
    use_facing: bool = True
    use_brightness: bool = True
    use_consistency: bool = True
    use_beats: bool = True
    use_motion: bool = True


@dataclass
class FusionResult:
    pulse: np.ndarray  # (T,) fused pulse, before the final clean-up
    window_centres: np.ndarray  # (W,) seconds
    weights: np.ndarray  # (W, P) normalised trust per patch per window (rows sum to 1)
    confidence: np.ndarray  # (W,) 0-1
    parts: np.ndarray  # (W, 4) confidence components, in CONFIDENCE_PARTS order
    patch_pulses: np.ndarray  # (T, P) cleaned POS pulse of every patch


def patch_pulses(rgb: np.ndarray, fs: float, valid: np.ndarray, s: Settings = SETTINGS) -> np.ndarray:
    """Cleaned (detrended, band-passed) POS pulse of every valid patch; zeros for the others."""
    n, n_p, _ = rgb.shape
    out = np.zeros((n, n_p))
    for k in np.flatnonzero(valid):
        out[:, k] = bandpass(detrend(pos(fill_gaps(rgb[:, k]), fs, s.pos_window_s), s.detrend_lambda), fs, s.hr_band)
    return out


@dataclass
class Motion:
    regressors: np.ndarray  # (T, 12) band-passed pose channels and their rates of change
    channel: np.ndarray  # (12,) pose channel (0-5) of each regressor column
    active: np.ndarray  # (T, 6) True where that pose channel really moves (see Settings.motion_min_*)

    def columns(self, sl: slice) -> np.ndarray | None:
        """Regressor columns to use in a window: those of channels moving in most of it, or None."""
        ch = np.flatnonzero(self.active[sl].mean(axis=0) >= 0.5)
        cols = np.flatnonzero(np.isin(self.channel, ch))
        return cols if len(cols) else None


def motion_regressors(pose: np.ndarray | None, fs: float, s: Settings = SETTINGS) -> Motion | None:
    """Band-passed head-pose signals, their rates of change, and where each really moves; None if unusable.

    Only channels that move clearly (rotation RMS >= ``motion_min_rot_deg``, translation RMS >=
    ``motion_min_trans_cm``) are used for removal. Small pose fluctuations are mostly estimation
    jitter, and MediaPipe's pose jitters *with the pulse* (the skin's brightness changes nudge the
    depth estimate; the head also bobs slightly with each beat), so regressing on them would
    subtract real pulse.
    """
    if pose is None or not np.isfinite(pose).any(axis=0).all():
        return None
    from scipy.ndimage import uniform_filter1d

    p = fill_gaps(np.asarray(pose, dtype=np.float64))
    bp = bandpass(detrend(p, s.detrend_lambda), fs, s.hr_band)  # physical units: deg, cm
    rms = np.sqrt(uniform_filter1d(bp**2, max(3, int(round(s.window_s * fs / 2))), axis=0, mode="nearest"))
    thr = np.array([s.motion_min_rot_deg] * 3 + [s.motion_min_trans_cm] * 3)
    x = np.concatenate([p, np.gradient(p, axis=0) * fs], axis=1)
    return Motion(regressors=bandpass(detrend(x, s.detrend_lambda), fs, s.hr_band),
                  channel=np.tile(np.arange(6), 2), active=rms >= thr[None, :])


def remove_motion_from_colour(rgb: np.ndarray, valid: np.ndarray, motion: Motion, fs: float,
                              s: Settings = SETTINGS) -> np.ndarray:
    """Subtract, from every patch's normalised colour trace, the part a ridge regression on head
    pose explains, in overlapping windows (Hann overlap-add).

    Done before POS because there the artifact is still a linear function of the motion; after
    POS's per-window alpha tuning it no longer is. Windows where the head does not really move
    are left untouched.
    """
    n, n_p, _ = rgb.shape
    out = rgb.copy()
    win = min(n, int(round(s.window_s * fs)))
    step = max(1, win // 2)
    starts = list(range(0, n - win + 1, step))
    if starts[-1] + win < n:
        starts.append(n - win)
    hann = np.hanning(win + 2)[1:-1]
    for k in np.flatnonzero(valid):
        c = fill_gaps(rgb[:, k])
        mean = c.mean(axis=0)
        cn = c / mean - 1.0
        y = bandpass(cn, fs, s.hr_band)
        corr, norm = np.zeros_like(cn), np.zeros(n)
        for s0 in starts:
            sl = slice(s0, s0 + win)
            norm[sl] += hann
            cols = motion.columns(sl)
            if cols is None:
                continue
            x = motion.regressors[sl][:, cols]
            x = x - x.mean(axis=0)
            x = x / np.maximum(x.std(axis=0), 1e-9)
            beta = np.linalg.solve(x.T @ x + s.motion_ridge * win * np.eye(x.shape[1]), x.T @ (y[sl] - y[sl].mean(axis=0)))
            corr[sl] += (x @ beta) * hann[:, None]
        out[:, k] = (1.0 + cn - corr / np.maximum(norm, 1e-9)[:, None]) * mean
    return out


def remove_motion(pulse: np.ndarray, regressors: np.ndarray, ridge: float) -> np.ndarray:
    """Subtract the ridge-regression fit of ``pulse`` on ``regressors`` (one window)."""
    x = regressors - regressors.mean(axis=0)
    x = x / np.maximum(x.std(axis=0), 1e-9)
    n, k = x.shape
    beta = np.linalg.solve(x.T @ x + ridge * n * np.eye(k), x.T @ pulse)
    return pulse - x @ beta


def rotation_speed(pose: np.ndarray | None, fs: float) -> np.ndarray | None:
    """Head rotation speed in deg/s per frame (from yaw, pitch, roll), or None."""
    if pose is None or not np.isfinite(pose[:, :3]).any():
        return None
    ang = fill_gaps(np.asarray(pose[:, :3], dtype=np.float64))
    return np.linalg.norm(np.gradient(ang, axis=0) * fs, axis=1)


def fuse(rgb: np.ndarray, fs: float, facing: np.ndarray | None = None, pose: np.ndarray | None = None,
         cfg: FusionConfig = FusionConfig(), s: Settings = SETTINGS) -> FusionResult:
    """Run steps 4-7 on patch colour traces ``rgb`` (T, P, 3)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    n, n_p, _ = rgb.shape
    valid = np.isfinite(rgb).all(axis=2).mean(axis=0) >= s.min_valid_fraction
    if not valid.any():
        raise ValueError("no patch is visible in enough frames")
    motion = motion_regressors(pose, fs, s) if cfg.use_motion else None
    if motion is not None and motion.active.any():
        rgb = remove_motion_from_colour(rgb, valid, motion, fs, s)
    pulses = patch_pulses(rgb, fs, valid, s)
    lum = fill_gaps(np.where(valid[None, :], luma(rgb), 128.0)) if cfg.use_brightness else None
    face = None
    if cfg.use_facing and facing is not None and np.isfinite(facing[:, valid]).any():
        face = fill_gaps(np.where(valid[None, :], facing, 1.0))
    speed = rotation_speed(pose, fs)

    win = min(n, int(round(s.window_s * fs)))
    step = max(1, int(round(s.step_s * fs)))
    starts = list(range(0, n - win + 1, step))
    if starts[-1] + win < n:
        starts.append(n - win)
    hann = np.hanning(win + 2)[1:-1]
    out, norm = np.zeros(n), np.zeros(n)
    centres, all_w, conf, parts = [], [], [], []
    for s0 in starts:
        sl = slice(s0, s0 + win)
        seg = pulses[sl]
        std = seg.std(axis=0)
        use = valid & (std > 0)
        seg = np.where(use, seg / np.where(std > 0, std, 1.0), 0.0)

        w0 = use.astype(np.float64)
        if face is not None:
            w0 = w0 * facing_weight(face[sl].mean(axis=0), s.min_facing)
        if lum is not None:
            w0 = w0 * brightness_weight(lum[sl].mean(axis=0), s.dark_luma, s.bright_luma)
        if w0.sum() <= 0:  # nothing passes the visual checks: keep the output defined
            w0 = use.astype(np.float64)
        cons = consistency(seg, w0 / w0.sum())
        w = w0 * cons**s.consistency_power if cfg.use_consistency else w0.copy()
        if cfg.use_beats:
            base = w if w.sum() > 0 else w0
            w = w * beat_agreement_scores(seg, base / base.sum(), fs, s.beat_tol_s)
        if w.sum() <= 0:
            w = w0.copy()
        w = w / w.sum()

        fused = seg @ w
        cols = motion.columns(sl) if motion is not None else None
        if cols is not None:  # second pass: what POS's per-window tuning made non-linear in the colours
            fused = remove_motion(fused, motion.regressors[sl][:, cols], s.motion_ridge)

        c_cons = float((w * cons).sum()) if n_p > 1 else 1.0
        c_shape = beat_shape_consistency(fused, fs)
        if speed is None:
            c_motion = 1.0
        else:
            c_motion = 1.0 / (1.0 + (float(np.mean(speed[sl])) / s.motion_ref_deg_s) ** 2)
        n_eff = 1.0 / float((w**2).sum())
        c_patch = min(1.0, n_eff / max(1.0, s.min_effective_patch_share * n_p))
        p = np.array([c_cons, c_shape, c_motion, c_patch])

        out[sl] += fused * hann
        norm[sl] += hann
        centres.append((s0 + win / 2) / fs)
        all_w.append(w)
        parts.append(p)
        conf.append(float(np.prod(np.clip(p, 0, 1)) ** (1.0 / len(p))))

    return FusionResult(pulse=out / np.maximum(norm, 1e-9), window_centres=np.array(centres), weights=np.array(all_w),
                        confidence=np.array(conf), parts=np.array(parts), patch_pulses=pulses)
