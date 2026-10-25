import numpy as np

from afscreen import synth
from afscreen.flags.ectopy import EctopyRule, detect_extra_beats, ibis_without_extra_beats
from afscreen.flags.experiment import hr_flag, score_record
from afscreen.flags.mitbih import _parse_txt

ANN = """      Time   Sample #  Type  Sub Chan  Num\tAux
    0:00.050       18     +    0    0    0\t(N
    0:00.500      180     N    0    0    0
    0:01.300      468     N    0    0    0
    0:02.000      720     A    0    0    0
    0:03.000     1080     ~    0    0    0
    0:03.100     1116     +    0    0    0\t(AFIB
    0:03.200     1152     N    0    0    0
"""


def test_parse_kaggle_annotations():
    rec = _parse_txt("201", ANN)
    assert list(rec.symbols) == ["N", "N", "A", "N"]  # rhythm and noise marks are not beats
    assert np.allclose(rec.times, [0.5, 1.3, 2.0, 3.2])
    assert list(rec.in_af) == [False, False, False, True]


def test_rule_finds_inserted_extra_beats_and_nothing_in_a_steady_rhythm():
    reg = synth.regular_beats(120, 70, seed=4)
    assert len(detect_extra_beats(reg)) <= 1
    extra = synth.with_extra_beats(reg, every=8, prematurity=0.35, seed=4)
    moved = np.flatnonzero(np.abs(extra - reg) > 1e-9)
    found = detect_extra_beats(extra)
    assert set(moved) <= set(found)
    assert len(set(found) - set(moved)) <= 1


def test_removal_drops_both_intervals_but_not_when_too_many_flagged():
    reg = synth.regular_beats(60, 70, seed=5)
    extra = synth.with_extra_beats(reg, every=10, seed=5)
    ibis, n, removed = ibis_without_extra_beats(extra)
    assert removed and n > 0 and len(ibis) == len(extra) - 1 - 2 * n
    _, _, removed_af = ibis_without_extra_beats(synth.af_beats(60, 90, seed=5), EctopyRule(max_share=0.01))
    assert not removed_af


def test_neighbour_check_spares_af_but_still_removes_isolated_extra_beats():
    rule = EctopyRule(max_share=1.0, neighbour_tol=0.1)
    af = synth.af_beats(120, 90, irregularity=0.3, seed=6)
    _, n_af, _ = ibis_without_extra_beats(af, EctopyRule(max_share=1.0))
    _, n_af_checked, _ = ibis_without_extra_beats(af, rule)
    assert n_af > 0 and n_af_checked < 0.3 * n_af
    extra = synth.with_extra_beats(synth.regular_beats(120, 70, seed=6), every=10, seed=6)
    _, n, removed = ibis_without_extra_beats(extra, rule)
    assert removed and n >= 0.8 * len(detect_extra_beats(extra))


def test_score_record_counts_outside_af_only():
    rec = _parse_txt("x", ANN)
    s = score_record(rec, EctopyRule())
    assert s["premature"] == 1 and s["beats"] == 3  # the beat in AF is not scored


def test_hr_flag():
    assert [hr_flag(x) for x in (45, 72, 120, np.nan)] == ["slow", "normal", "fast", "insufficient"]
