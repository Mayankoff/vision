"""Synthetic data with *known beat times*, for tests and for end-to-end checks without datasets.

Rhythms: regular (with breathing-linked variation), AF-like (random intervals), slow, fast, and
regular with inserted extra (premature) beats. The pulse is built beat by beat, so beat timing
is exact, and colour follows the skin-reflection model of Wang et al. (2017):

    C(t) = I(t) * (u_skin + strength * p(t) * u_pulse) + noise
"""

from __future__ import annotations

import cv2
import numpy as np

SKIN_RGB = np.array([0.70, 0.50, 0.40])  # typical normalised skin reflectance
PULSE_RGB = np.array([0.33, 0.77, 0.53])  # blood-volume-pulse colour signature, strongest in green


# ----------------------------------------------------------------------------- rhythms


def regular_beats(duration_s: float, hr_bpm: float, rsa_ms: float = 30.0, breath_hz: float = 0.25,
                  jitter_ms: float = 8.0, seed: int = 0) -> np.ndarray:
    """Normal rhythm: intervals vary smoothly with breathing (sinus arrhythmia) plus small noise."""
    rng = np.random.default_rng(seed)
    beats, t = [], rng.uniform(0, 60.0 / hr_bpm)
    while t < duration_s:
        beats.append(t)
        ibi = 60.0 / hr_bpm + rsa_ms / 1000 * np.sin(2 * np.pi * breath_hz * t) + rng.normal(0, jitter_ms / 1000)
        t += max(ibi, 0.25)
    return np.array(beats)


def af_beats(duration_s: float, mean_hr_bpm: float = 90.0, irregularity: float = 0.3, seed: int = 0) -> np.ndarray:
    """AF-like rhythm: independent, random intervals (+/- ``irregularity`` of the mean), no pattern."""
    rng = np.random.default_rng(seed)
    mean = 60.0 / mean_hr_bpm
    beats, t = [], rng.uniform(0, mean)
    while t < duration_s:
        beats.append(t)
        t += max(0.3, mean * (1 + irregularity * rng.uniform(-1, 1)))
    return np.array(beats)


def with_extra_beats(beats: np.ndarray, every: int = 8, prematurity: float = 0.35, seed: int = 0) -> np.ndarray:
    """Make every ``every``-th beat premature: it comes ``prematurity`` x interval early and the next
    beat stays on time, giving the short-long pattern of an extra beat. Returns the new beat times."""
    rng = np.random.default_rng(seed)
    b = np.array(beats, dtype=np.float64)
    start = int(rng.integers(2, every + 2))
    for i in range(start, len(b) - 1, every):
        b[i] = b[i - 1] + (1 - prematurity) * (b[i] - b[i - 1])
    return b


def pulse_from_beats(t: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance PPG-like waveform with a systolic peak at each beat time."""
    beats = np.asarray(beats, dtype=np.float64)
    ibis = np.diff(beats, append=beats[-1] + (beats[-1] - beats[-2] if len(beats) > 1 else 1.0))
    x = np.zeros_like(t, dtype=np.float64)
    for b, m in zip(beats, ibis):
        # The wave's timing scales only weakly with the interval: the diastolic (dicrotic) bump stays
        # ~0.3 s after the systolic peak, as in real pulses, even after a long pause.
        f = np.clip(m, 0.45, 0.9) / 0.8
        near = (t > b - 1.0) & (t < b + 1.5)
        tt = t[near] - b
        x[near] += np.exp(-0.5 * (tt / (0.09 * f)) ** 2) + 0.35 * np.exp(-0.5 * ((tt - 0.28 * f) / (0.09 * f)) ** 2)
    x -= x.mean()
    return x / x.std()


# ------------------------------------------------------------------------ patch traces


def patch_traces(beats: np.ndarray, fs: float = 30.0, duration_s: float = 30.0, n_patches: int = 25,
                 strength: float = 0.003, noise: float = 0.0008, noisy: tuple[int, ...] = (), noisy_level: float = 0.02,
                 dark: tuple[int, ...] = (), turned: tuple[int, ...] = (), motion_hz: float | None = None,
                 motion_deg: float = 6.0, motion_artifact: float = 0.004, seed: int = 0) -> dict:
    """Per-patch RGB traces (T, P, 3) in 0-255, with optional problem patches and head motion.

    ``noisy`` patches get strong independent noise, ``dark`` patches sit in shadow, ``turned``
    patches face away from the camera. With ``motion_hz`` the head turns (yaw at ``motion_hz``,
    pitch at 1.37 x ``motion_hz``) and tinted light makes every patch's colour follow each motion
    in a different colour direction. POS can cancel one distortion direction, not two, so the
    artifact survives into the pulse unless motion removal takes it out.
    Returns dict(rgb, facing, pose, pulse, beats, fs).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * fs)) / fs
    p = pulse_from_beats(t, beats)
    intensity = 1.0 + 0.02 * np.sin(2 * np.pi * t / duration_s)
    base = intensity[:, None, None] * (SKIN_RGB + strength * p[:, None, None] * PULSE_RGB)
    rgb = np.repeat(base, n_patches, axis=1)
    rgb = rgb * rng.uniform(0.9, 1.1, (1, n_patches, 1))  # patches differ a little in tone
    rgb += noise * rng.standard_normal(rgb.shape)
    for k in noisy:
        rgb[:, k] += noisy_level * rng.standard_normal((len(t), 3))
    pose = np.zeros((len(t), 6))
    if motion_hz:
        tints = (np.array([1.0, 0.4, -0.6]), np.array([-0.3, 0.8, 0.2]))  # two lamps of different colour
        for axis, (f, tint) in enumerate(zip((motion_hz, 1.37 * motion_hz), tints)):
            angle = motion_deg * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
            pose[:, axis] = angle
            gain = rng.uniform(0.6, 1.4, n_patches)
            rgb += motion_artifact * (angle / motion_deg)[:, None, None] * gain[None, :, None] * tint
    rgb *= 255.0
    for k in dark:
        rgb[:, k] *= 0.06
    facing = np.full((len(t), n_patches), 0.95)
    for k in turned:
        facing[:, k] = 0.15
    return dict(rgb=rgb, facing=facing, pose=pose, pulse=p, beats=np.asarray(beats), fs=fs)


# ------------------------------------------------------------------------------ videos


def face_video(face_rgb: np.ndarray, skin_mask: np.ndarray, beats: np.ndarray, fps: float = 30.0,
               duration_s: float = 20.0, strength: float = 0.012, jitter_px: float = 0.8, noise_std: float = 1.0,
               scale: float = 1.0, sway_hz: float | None = None, sway_px: float = 6.0, seed: int = 0):
    """Yield frames of a still face photo whose skin pulses with the given beat times.

    ``scale`` < 1 shrinks the face inside a frame of the same size (small faces, far from the
    camera). With ``sway_hz`` the face moves side to side under a tinted light gradient, so the
    skin colour changes with head position (a motion artifact).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    t = np.arange(n) / fps
    p = pulse_from_beats(t, beats)
    h, w = face_rgb.shape[:2]
    base, mask = face_rgb.astype(np.float32), skin_mask
    if scale != 1.0:
        small = cv2.resize(face_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        small_mask = cv2.resize(skin_mask.astype(np.uint8), (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        base = np.full_like(base, base.reshape(-1, 3).mean(axis=0))
        mask = np.zeros_like(skin_mask)
        y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
        base[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        mask[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small_mask
    gain = (1.0 + strength * p[:, None] * (PULSE_RGB / PULSE_RGB.mean())[None, :]).astype(np.float32)
    ramp = np.linspace(-1, 1, w, dtype=np.float32)[None, :, None] * np.array([0.10, 0.03, -0.06], np.float32)
    phase = rng.uniform(0, 2 * np.pi)
    for i in range(n):
        frame = base.copy()
        frame[mask] *= gain[i]
        dx, dy = rng.normal(0, jitter_px, 2)
        if sway_hz:
            dx += sway_px * np.sin(2 * np.pi * sway_hz * t[i] + phase)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        if sway_hz:
            frame *= 1.0 + ramp  # fixed light gradient: moving skin passes through it
        frame += rng.normal(0, noise_std, frame.shape).astype(np.float32)
        yield np.clip(frame, 0, 255).astype(np.uint8)


def skin_mask_from_photo(face_rgb: np.ndarray, tracker=None) -> np.ndarray:
    """Convex hull of the face landmarks of a photo (where the synthetic pulse is applied)."""
    from afscreen.face.tracker import FaceTracker

    own = tracker is None
    tracker = tracker or FaceTracker(mode="image")
    try:
        face = tracker.detect(face_rgb)
    finally:
        if own:
            tracker.close()
    if face is None:
        raise ValueError("no face found in the photo")
    mask = np.zeros(face_rgb.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(face.points[:468, :2]).astype(np.int32)), 1)
    return mask.astype(bool)
