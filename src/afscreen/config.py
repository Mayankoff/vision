"""Settings shared by every stage, so experiments are comparable.

Change a value only deliberately, and re-run every experiment if you do.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Heart-rate band in Hz: 0.5-3.3 Hz = 30-200 BPM. Wider than the usual 42-180 BPM so that
    # slow (<50 BPM) and fast (>100 BPM) heart rates can be measured at all (secondary output).
    hr_band: tuple[float, float] = (0.5, 3.3)
    # Butterworth band-pass order; applied forward-backward (zero phase), so effectively doubled.
    filter_order: int = 2
    # Smoothness-priors detrending strength (Tarvainen et al., 2002).
    detrend_lambda: float = 100.0
    # POS sliding-window length in seconds (Wang et al., 2017).
    pos_window_s: float = 1.6
    # Minimum FFT length (zero-padding) so the HR grid is finer than ~0.1 BPM at 30 fps.
    min_nfft: int = 2**15

    # Trust scores, fusion and confidence are computed in windows of this length, updated every step.
    window_s: float = 8.0
    step_s: float = 1.0

    # Patches: the finest patch level whose patches have at least this many pixels is used.
    min_patch_pixels: float = 350.0

    # Trust checks
    min_facing: float = 0.3  # |cos| of the angle to the camera below which a patch gets weight 0 (~72 deg)
    dark_luma: tuple[float, float] = (20.0, 45.0)  # weight ramps 0 -> 1 across this luma range
    bright_luma: tuple[float, float] = (220.0, 245.0)  # weight ramps 1 -> 0 across this luma range
    consistency_power: float = 2.0  # sharpens the difference between consistent and inconsistent patches
    beat_tol_s: float = 0.10  # a patch beat "agrees" when within this time of a consensus beat

    # Motion removal: ridge strength (x window length) of the head-pose regression.
    motion_ridge: float = 0.1
    # Motion removal runs only in windows where the head really moves: band-passed rotation RMS
    # (deg) or translation RMS (cm) at least this large. Below it, pose "motion" is mostly
    # estimation jitter, and MediaPipe's pose jitters *with the pulse* (skin brightness changes
    # nudge the depth estimate), so regressing on it would subtract real pulse.
    motion_min_rot_deg: float = 1.0
    motion_min_trans_cm: float = 0.5
    # Confidence: head rotation speed (deg/s) at which the motion component of the confidence is 0.5.
    motion_ref_deg_s: float = 5.0
    # Confidence: effective number of trusted patches (as a share of all patches) that counts as "enough".
    min_effective_patch_share: float = 0.3

    # Patches visible in fewer frames than this share are dropped.
    min_valid_fraction: float = 0.5


SETTINGS = Settings()
