import zipfile

import numpy as np
import pandas as pd
import pytest

from afscreen import synth
from afscreen.af.mimic import load_mimic_testing
from afscreen.signal.beats import detect_beats
from afscreen.signal.filters import clean_pulse
from afscreen.signal.hrv import rmssd_ms, sdnn_ms
from afscreen.signal.resp import breathing_rate, reference_rate
from afscreen.signal.spectrum import hr_fft, hr_fft_breathing_aware

FS = 30.0


def _breathing_pulse(br_bpm, hr_bpm=72, duration=64, seed=1):
    t = np.arange(int(duration * FS)) / FS
    beats = synth.regular_beats(duration, hr_bpm, rsa_ms=40, breath_hz=br_bpm / 60, jitter_ms=2, seed=seed)
    x = synth.pulse_from_beats(t, beats) * (1 + 0.2 * np.sin(2 * np.pi * br_bpm / 60 * t))
    return t, clean_pulse(x, FS)


@pytest.mark.parametrize("br", [8, 12, 18, 24])
def test_breathing_rate_from_amplitude_and_interval_modulation(br):
    _, x = _breathing_pulse(br)
    rate, est = breathing_rate(x, FS, detect_beats(x, FS))
    assert abs(rate - br) < 1.0 and set(est) == {"RIAV", "RIFV"}


def test_irregular_rhythm_drops_rifv():
    _, x = _breathing_pulse(15)
    _, est = breathing_rate(x, FS, detect_beats(x, FS), irregular=True)
    assert "RIFV" not in est


def test_reference_rate_needs_spectrum_and_count_to_agree():
    t = np.arange(int(64 * 125)) / 125
    assert abs(reference_rate(np.sin(2 * np.pi * 0.25 * t), 125) - 15) < 0.5
    noise = np.random.default_rng(0).normal(size=len(t))
    assert np.isnan(reference_rate(noise, 125)) or reference_rate(noise, 125) > 0  # never crashes


def test_breathing_aware_hr_skips_a_breathing_harmonic():
    t = np.arange(int(30 * FS)) / FS
    br = 15.0  # 0.25 Hz; its 3rd harmonic (45 /min) lies inside the 30-200 BPM band
    x = 1.0 * np.sin(2 * np.pi * 0.75 * t) + 0.6 * np.sin(2 * np.pi * 1.2 * t)  # 45 (harmonic) beats 72 (pulse)
    assert abs(hr_fft(x, FS) - 45) < 1
    assert abs(hr_fft_breathing_aware(x, FS, br) - 72) < 1
    assert abs(hr_fft_breathing_aware(x, FS, float("nan")) - 45) < 1  # unknown breathing: plain FFT


def test_hrv():
    ibis = np.array([0.8, 0.82, 0.8, 0.82, 0.8])
    assert abs(rmssd_ms(ibis) - 20) < 1e-6
    assert abs(sdnn_ms(ibis) - np.std(ibis, ddof=1) * 1000) < 1e-9
    assert np.isnan(rmssd_ms(ibis[:2]))


@pytest.mark.parametrize("refine", ["none", "parabolic", "spline"])
def test_beat_refinement_options(refine):
    t = np.arange(int(30 * FS)) / FS
    beats = synth.regular_beats(30, 72, seed=2)
    x = clean_pulse(synth.pulse_from_beats(t, beats), FS)
    b = detect_beats(x, FS, refine=refine)
    err = np.abs(np.diff(b) - np.diff(beats[(beats >= b[0] - 0.1) & (beats <= b[-1] + 0.1)])[:len(b) - 1])
    assert np.median(err) < (0.02 if refine == "none" else 0.01)


def test_load_mimic_testing_reads_resp_and_group(tmp_path):
    t = np.arange(1250) / 125
    df = pd.DataFrame({"Time": t, "PPG": np.sin(t), "ECG": 0.0, "resp": np.cos(t)})
    with zipfile.ZipFile(tmp_path / "mimic_perform_test_all_csv.zip", "w") as zf:
        for i, g in enumerate("an"):
            zf.writestr(f"mimic_perform_test_all_csv/mimic_perform_test_all_00{i}_data.csv", df.to_csv(index=False))
            zf.writestr(f"mimic_perform_test_all_csv/mimic_perform_test_all_00{i}_fix.txt", f"x\nGroup: {g}\n")
    (adult,) = load_mimic_testing(tmp_path)
    assert adult.group == "a" and adult.resp is not None and adult.fs == 125.0
    assert len(load_mimic_testing(tmp_path, None)) == 2
