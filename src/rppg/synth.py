"""Synthetic signals and videos with a known heart rate, for tests and smoke runs.

The RGB model follows the skin-reflection model of Wang et al. (2017):

    C(t) = I(t) * (u_skin + pulse_strength * p(t) * u_pulse) + noise

where I(t) is the illumination intensity (drift + optional flicker), u_skin the
skin colour and u_pulse the blood-volume-pulse colour signature (the PBV
vector of de Haan & van Leest, 2014). Intensity changes affect all channels
proportionally, which is exactly what CHROM and POS are designed to cancel.
"""

from __future__ import annotations

import cv2
import numpy as np

SKIN_RGB = np.array([0.70, 0.50, 0.40])  # typical normalised skin reflectance
PULSE_RGB = np.array([0.33, 0.77, 0.53])  # BVP signature, strongest in green


def pulse_wave(t: np.ndarray, hr_bpm: float, hrv: float = 0.0, rng=None) -> np.ndarray:
    """Zero-mean PPG-like waveform (fundamental + 2nd harmonic), optional slow HR variation."""
    rng = np.random.default_rng(rng)
    f = hr_bpm / 60.0
    phase_noise = hrv * np.cumsum(rng.standard_normal(len(t))) / max(1, len(t)) ** 0.5
    phase = 2 * np.pi * f * t + phase_noise
    w = np.sin(phase) + 0.35 * np.sin(2 * phase - 0.8)
    return w / w.std()


def synthetic_rgb_trace(
    hr_bpm: float = 72.0,
    fs: float = 30.0,
    duration_s: float = 30.0,
    pulse_strength: float = 0.002,
    noise: float = 0.0005,
    drift: float = 0.02,
    flicker_hz: float | None = None,
    flicker_strength: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, rgb (T, 3) in 0-255 range, ground-truth pulse (T,))."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * fs)) / fs
    p = pulse_wave(t, hr_bpm, rng=rng)
    intensity = 1.0 + drift * np.sin(2 * np.pi * t / duration_s)  # slow auto-exposure-like drift
    if flicker_hz:
        intensity = intensity + flicker_strength * np.sin(2 * np.pi * flicker_hz * t)
    rgb = intensity[:, None] * (SKIN_RGB[None, :] + pulse_strength * p[:, None] * PULSE_RGB[None, :])
    rgb = rgb + noise * rng.standard_normal(rgb.shape)
    return t, rgb * 255.0, p


def synthetic_face_video(
    face_rgb: np.ndarray,
    skin_mask: np.ndarray,
    hr_bpm: float = 72.0,
    fps: float = 30.0,
    duration_s: float = 20.0,
    pulse_strength: float = 0.01,
    jitter_px: float = 1.5,
    noise_std: float = 1.0,
    seed: int = 0,
):
    """Yield frames of a still face photo whose skin pixels pulse at ``hr_bpm``.

    Small random translations and sensor noise are added so the pipeline has
    to track the face rather than read a perfectly static image. Returns a
    generator; the ground-truth pulse is available via :func:`pulse_wave`
    with the same arguments.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    t = np.arange(n) / fps
    p = pulse_wave(t, hr_bpm, rng=np.random.default_rng(seed))
    base = face_rgb.astype(np.float32)
    gain = (1.0 + pulse_strength * p[:, None] * (PULSE_RGB / PULSE_RGB.mean())[None, :]).astype(np.float32)
    h, w = face_rgb.shape[:2]
    for i in range(n):
        frame = base.copy()
        frame[skin_mask] *= gain[i]
        dx, dy = rng.normal(0, jitter_px, 2)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        frame += rng.normal(0, noise_std, frame.shape).astype(np.float32)
        yield np.clip(frame, 0, 255).astype(np.uint8)
