"""Frozen evaluation protocol (Phase 0).

Every method - our classical pipeline, toolbox unsupervised methods and toolbox
deep models - is scored with these settings. Toolbox outputs are re-scored with
this module (see Phase 2) rather than with the toolbox's own post-processing,
whose band (0.6-3.3 Hz at the pinned commit) is hard-coded and differs from ours.

Change a value here only deliberately, and re-run every experiment if you do.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    # Heart-rate band in Hz: 0.7-3.0 Hz = 42-180 BPM (as stated in the Review 2 report).
    hr_band: tuple[float, float] = (0.7, 3.0)
    # Butterworth band-pass order; applied forward-backward (filtfilt), so the
    # effective order is doubled and there is no phase shift.
    filter_order: int = 2
    # Smoothness-priors detrending strength (Tarvainen et al. 2002), as used by the toolbox.
    detrend_lambda: float = 100.0
    # Primary HR estimator: "fft" (periodogram peak). "peak" is reported as a secondary column.
    hr_method: str = "fft"
    # Minimum FFT length (zero-padding) so the HR grid is finer than ~0.1 BPM at 30 fps.
    min_nfft: int = 2**15
    # Evaluation window. None = one HR per whole video (primary table);
    # a number = HR per window of that many seconds (secondary table).
    window_s: float | None = None
    window_step_s: float = 1.0
    # SNR (de Haan & Jeanne 2013): power within +/- this many BPM of the GT HR
    # fundamental and first harmonic, versus the rest of the HR band.
    snr_tolerance_bpm: float = 6.0
    # POS / CHROM sliding-window length in seconds (from the original papers).
    projection_window_s: float = 1.6
    # Ground-truth HR is computed from the reference BVP with the same estimator
    # and window as the prediction (not the oximeter's displayed HR).
    gt_from_bvp: bool = True


PROTOCOL = Protocol()
