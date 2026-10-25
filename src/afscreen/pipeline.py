"""Steps 2-7 on cached traces, plus heart rate and beats: one call per video and variant.

Variants form the ablation ladder of the plan (Phase 5); each adds one ingredient:

    single        one skin region (all patches merged), plain POS - the baseline
    patch_mean    patches, equal weights
    +facing       + face-orientation check
    +brightness   + brightness check
    +consistency  + patch-consistency check
    +beats        + beat-agreement check
    full          + motion removal (the proposed method)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from afscreen.config import SETTINGS, Settings
from afscreen.core.fusion import FusionConfig, FusionResult, fuse
from afscreen.face.patches import choose_level
from afscreen.signal.beats import detect_beats
from afscreen.signal.filters import clean_pulse
from afscreen.signal.spectrum import hr_fft
from afscreen.traces import PatchTraces

_OFF = dict(use_facing=False, use_brightness=False, use_consistency=False, use_beats=False, use_motion=False)
VARIANTS: dict[str, tuple[str, FusionConfig]] = {
    "single": ("single", FusionConfig(**_OFF)),
    "patch_mean": ("auto", FusionConfig(**_OFF)),
    "+facing": ("auto", FusionConfig(**{**_OFF, "use_facing": True})),
    "+brightness": ("auto", FusionConfig(**{**_OFF, "use_facing": True, "use_brightness": True})),
    "+consistency": ("auto", FusionConfig(use_beats=False, use_motion=False)),
    "+beats": ("auto", FusionConfig(use_motion=False)),
    "full": ("auto", FusionConfig()),
}
# Comparison baselines (Phase 5): other pulse-extraction methods on one skin region.
BASELINES: dict[str, tuple[str, FusionConfig]] = {
    "single_green": ("single", FusionConfig(**_OFF, pulse="green")),
    "single_chrom": ("single", FusionConfig(**_OFF, pulse="chrom")),
}
ALL_VARIANTS = {**BASELINES, **VARIANTS}


@dataclass
class Result:
    variant: str
    level: int
    fps: float
    pulse: np.ndarray  # (T,) cleaned fused pulse, unit variance
    hr_bpm: float  # whole-video FFT heart rate
    window_t: np.ndarray  # (W,) window centres, s
    window_hr: np.ndarray  # (W,) FFT heart rate per window
    confidence: np.ndarray  # (W,)
    beats: np.ndarray  # beat times, s
    fusion: FusionResult

    @property
    def hr_beats_bpm(self) -> float:
        """60 / mean beat interval. Unlike the FFT peak, valid for an irregular rhythm too."""
        return float(60.0 / np.mean(np.diff(self.beats))) if len(self.beats) > 2 else float("nan")

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidence)) if len(self.confidence) else float("nan")

    wide_pulse: np.ndarray | None = None  # (T,) wide-band pulse for breathing (see wide_band_pulse)


def window_hr(pulse: np.ndarray, fs: float, centres: np.ndarray, s: Settings = SETTINGS) -> np.ndarray:
    """FFT heart rate of the ``s.window_s`` window around each centre."""
    win = min(len(pulse), int(round(s.window_s * fs)))
    out = []
    for c in centres:
        a = int(np.clip(round(c * fs - win / 2), 0, len(pulse) - win))
        out.append(hr_fft(pulse[a:a + win], fs, s.hr_band))
    return np.array(out)


def wide_band_pulse(rgb: np.ndarray, fs: float, weights: np.ndarray, band=(0.1, 3.3)) -> np.ndarray:
    """Trust-weighted, inverted green-channel pulse, band-passed to keep slow (breathing) changes.

    POS removes everything slower than its 1.6 s window, which takes the breathing-induced
    intensity changes (RIIV) with it. The normalised green channel keeps them; it is inverted
    because more blood in the skin means less reflected light. ``weights`` (P,) are the patch
    trust scores averaged over the video.
    """
    from afscreen.signal.filters import bandpass, fill_gaps

    valid = np.isfinite(rgb[:, :, 1]).mean(axis=0) >= 0.5
    w = np.where(valid, weights, 0.0)
    if w.sum() <= 0:
        w = valid.astype(np.float64)
    g = fill_gaps(rgb[:, valid, 1])
    g = g / g.mean(axis=0) - 1.0
    x = -(g @ (w[valid] / w[valid].sum()))
    y = bandpass(x - x.mean(), fs, band)
    return y / y.std() if y.std() > 0 else y


def analyse(traces: PatchTraces, variant: str = "full", level: int | None = None, s: Settings = SETTINGS) -> Result:
    """Fuse the patches of one video with ``variant`` and estimate heart rate and beats.

    ``level`` forces a patch level (25, 12, 6); by default it is chosen from the face size.
    """
    mode, cfg = ALL_VARIANTS[variant]
    lvl = 1 if mode == "single" else (level or choose_level(traces.counts, s.min_patch_pixels))
    rgb, _, facing = traces.at_level(lvl)
    fr = fuse(rgb, traces.fps, facing, traces.pose, cfg, s)
    pulse = clean_pulse(fr.pulse, traces.fps, s.hr_band)
    return Result(
        variant=variant, level=lvl, fps=traces.fps, pulse=pulse, hr_bpm=hr_fft(pulse, traces.fps, s.hr_band),
        window_t=fr.window_centres, window_hr=window_hr(pulse, traces.fps, fr.window_centres, s),
        confidence=fr.confidence, beats=detect_beats(pulse, traces.fps, s.hr_band), fusion=fr,
        wide_pulse=wide_band_pulse(rgb, traces.fps, fr.weights.mean(axis=0)),
    )
