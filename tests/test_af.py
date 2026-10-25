import io
import zipfile

import numpy as np
import pandas as pd

from afscreen import synth
from afscreen.af.experiment import build_table, evaluate, resample
from afscreen.af.features import FEATURES, clean_ibis, ibi_features, sample_entropy, tpr
from afscreen.af.mimic import load_mimic_af


def _feat(beats):
    return dict(zip(FEATURES, ibi_features(np.diff(beats))))


def test_af_scores_more_irregular_than_regular():
    reg = _feat(synth.regular_beats(30, 72, seed=1))
    af = _feat(synth.af_beats(30, 90, seed=1))
    assert af["nrmssd"] > 3 * reg["nrmssd"]
    assert af["sd1_sd2"] > reg["sd1_sd2"]
    assert af["tpr"] > reg["tpr"]
    # (sample entropy is not compared: on ~30 intervals it is too noisy to order single windows)


def test_feature_helpers():
    assert np.array_equal(clean_ibis([0.1, 0.8, 2.5, 1.0]), [0.8, 1.0])
    assert ibi_features(np.full(5, 0.8)) is None  # too few intervals
    assert tpr(np.array([1, 2, 1, 2, 1.0])) == 1.0
    assert sample_entropy(np.ones(20)) == 0.0


def test_resample_to_camera_rate():
    x = np.sin(2 * np.pi * 1.2 * np.arange(1250) / 125)
    assert len(resample(x, 125.0, 30.0)) == 300


def _csv(beats, fs=125.0, duration=120.0):
    t = np.arange(int(duration * fs)) / fs
    return pd.DataFrame({"Time": t, "PPG": synth.pulse_from_beats(t, beats), "ECG": 0.0}).to_csv(index=False)


def test_mimic_loader_table_and_gating(tmp_path):
    zf_af = zipfile.ZipFile(tmp_path / "mimic_perform_af_csv.zip", "w")
    zf_non = zipfile.ZipFile(tmp_path / "mimic_perform_non_af_csv.zip", "w")
    for i in range(3):
        zf_af.writestr(f"mimic_perform_af_csv/mimic_perform_af_00{i}_data.csv", _csv(synth.af_beats(120, 90, seed=i)))
        zf_non.writestr(f"mimic_perform_non_af_csv/mimic_perform_non_af_00{i}_data.csv",
                        _csv(synth.regular_beats(120, 70, seed=i)))
    zf_af.close()
    zf_non.close()
    recs = load_mimic_af(tmp_path)
    assert len(recs) == 6 and sum(r.af for r in recs) == 3 and recs[0].fs == 125.0
    df = build_table(recs, 30.0, camera_fs=30.0)
    assert len(df) == 24 and set(df.af) == {0, 1}
    res = evaluate(df, thresholds=(0.0,), models=("logreg",))
    assert res.loc[0, "logreg_subj_sens"] == 1.0 and res.loc[0, "logreg_subj_spec"] == 1.0
