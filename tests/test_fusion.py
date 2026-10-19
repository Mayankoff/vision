import numpy as np
import pytest

from afscreen import synth
from afscreen.core.fusion import FusionConfig, fuse
from afscreen.core.trust import beat_agreement_scores, brightness_weight, consistency, facing_weight
from afscreen.eval.metrics import ibi_error_ms
from afscreen.pipeline import VARIANTS, analyse
from afscreen.signal.filters import bandpass
from afscreen.traces import PatchTraces

FS, DUR = 30.0, 40.0


def traces(beats, **kw) -> tuple[PatchTraces, dict]:
    d = synth.patch_traces(beats, FS, DUR, seed=7, **kw)
    n = d["rgb"].shape[0]
    return PatchTraces(fps=FS, rgb=d["rgb"], counts=np.full((n, 25), 500.0), facing=d["facing"], pose=d["pose"]), d


def test_visual_weights():
    assert np.allclose(facing_weight(np.array([1.0, 0.5, 0.2, np.nan])), [1.0, 0.5, 0.0, 0.0])
    assert np.allclose(brightness_weight(np.array([5.0, 32.5, 120.0, 250.0])), [0.0, 0.5, 1.0, 0.0])


def test_consistency_and_beat_agreement_flag_the_odd_patch():
    t = np.arange(240) / FS
    pulse = synth.pulse_from_beats(t, synth.regular_beats(8, 72, seed=1))
    rng = np.random.default_rng(0)
    seg = np.stack([pulse + 0.2 * rng.standard_normal(len(t)) for _ in range(6)], axis=1)
    seg[:, 5] = rng.standard_normal(len(t))
    seg = bandpass(seg, FS)  # patch pulses are always band-passed in the pipeline
    seg /= seg.std(axis=0)
    w = np.full(6, 1 / 6)
    c = consistency(seg, w)
    assert c[5] < 0.2 and c[:5].min() > 0.8
    a = beat_agreement_scores(seg, w, FS)
    assert a[5] < 0.6 and a[:5].min() > 0.9


def test_problem_patches_get_low_trust():
    tr, _ = traces(synth.regular_beats(DUR, 80, seed=3), noisy=(1, 2), dark=(3,), turned=(4,))
    w = fuse(tr.rgb, FS, tr.facing, tr.pose).weights.mean(axis=0)
    good = np.delete(w, [1, 2, 3, 4])
    assert w[[1, 2, 3, 4]].max() < 0.2 * good.min()


def test_full_method_beats_single_region_on_noisy_patches():
    beats = synth.regular_beats(DUR, 66, seed=2)
    tr, d = traces(beats, noisy=tuple(range(0, 25, 3)))
    single = ibi_error_ms(analyse(tr, "single").beats, d["beats"])
    full = ibi_error_ms(analyse(tr, "full").beats, d["beats"])
    assert full < 0.5 * single


@pytest.mark.parametrize("motion_hz", [0.8, 1.3])
def test_motion_removal(motion_hz):
    beats = synth.regular_beats(DUR, 72, seed=4)
    tr, d = traces(beats, motion_hz=motion_hz, motion_artifact=0.02, motion_deg=3)
    gt = 60 / np.mean(np.diff(beats))
    no_motion = analyse(tr, "+beats")
    full = analyse(tr, "full")
    assert abs(full.hr_bpm - gt) < 1.0
    assert ibi_error_ms(full.beats, d["beats"]) < 0.5 * ibi_error_ms(no_motion.beats, d["beats"])


def test_pulse_locked_pose_jitter_is_not_removed():
    """MediaPipe's pose jitters with the pulse; small jitter must not trigger motion removal."""
    beats = synth.regular_beats(DUR, 72, seed=5)
    tr, d = traces(beats)
    tr.pose[:, 5] = 0.1 * d["pulse"] + 0.05 * np.random.default_rng(1).standard_normal(len(d["pulse"]))
    a = ibi_error_ms(analyse(tr, "+beats").beats, d["beats"])
    b = ibi_error_ms(analyse(tr, "full").beats, d["beats"])
    assert b <= a + 1.0


def test_af_keeps_high_confidence():
    tr, _ = traces(synth.af_beats(DUR, 95, irregularity=0.35, seed=6))
    assert analyse(tr, "full").mean_confidence > 0.85


def test_confidence_drops_when_patches_are_noise():
    tr, _ = traces(synth.regular_beats(DUR, 72, seed=8), strength=0.0003, noise=0.004)
    clean, _ = traces(synth.regular_beats(DUR, 72, seed=8))
    assert analyse(tr, "full").mean_confidence < analyse(clean, "full").mean_confidence - 0.2


def test_every_variant_runs_and_single_uses_one_region():
    tr, _ = traces(synth.regular_beats(DUR, 72, seed=9))
    for v in VARIANTS:
        r = analyse(tr, v)
        assert r.level == (1 if v == "single" else 25)
        assert np.isfinite(r.hr_bpm) and len(r.confidence) == len(r.window_t)
        assert np.allclose(r.fusion.weights.sum(axis=1), 1.0)


def test_disabled_checks_give_equal_weights():
    tr, _ = traces(synth.regular_beats(DUR, 72, seed=9), noisy=(0,))
    off = FusionConfig(use_facing=False, use_brightness=False, use_consistency=False, use_beats=False, use_motion=False)
    w = fuse(tr.rgb, FS, tr.facing, tr.pose, off).weights
    assert np.allclose(w, 1 / 25)
