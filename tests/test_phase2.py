import numpy as np
import pandas as pd
import pytest
from phase2_report import cross_check, load, roi_ablation
from score_predictions import video_identity

from rppg.io.datasets import mmpd_canonical_labels


def test_toolbox_names_map_to_our_ids():
    assert video_identity("UBFC-rPPG", "subject12") == ("subject12", "12", {})
    assert video_identity("PURE", "101")[:2] == ("01-01", "01")
    assert video_identity("PURE", "1006")[:2] == ("10-06", "10")
    vid, subject, labels = video_identity("MMPD", "subject5_L4_MO1_E1_S5_GE2_GL2_H1_MA2")
    assert subject == "5" and labels["light"] == "Nature" and labels["motion"] == "Stationary"
    assert labels["skin_color"] == "5" and labels["hair_cover"] == "True"


def _row(**kw):
    base = {"dataset": "UBFC-rPPG", "hr_pred_peak": np.nan, "hr_gt_peak": np.nan, "snr_db": 1.0, "method_ms_per_frame": 0.1}
    return {**base, **kw}


def test_mmpd_videos_match_across_codebases(tmp_path):
    ours_labels = mmpd_canonical_labels({"light": "Nature", "motion": "Stationary (after exercise)", "exercise": "True",
                                         "skin_color": "5", "gender": "female", "glasser": "False", "hair_cover": "True",
                                         "makeup": "False"})
    _, _, tb_labels = video_identity("MMPD", "subject5_L4_MO1_E1_S5_GE2_GL2_H1_MA2")
    assert ours_labels == tb_labels
    lab = lambda d: {f"label_{k}": v for k, v in d.items()}  # noqa: E731
    ours = pd.DataFrame([
        _row(dataset="MMPD", id="p5_3", subject="5", impl="ours", method="pos", roi=r, hr_pred=p, hr_gt=70.0, **lab(ours_labels))
        for r, p in (("skin", 71.0), ("face_box", 75.0))
    ] + [_row(id="subject1", subject="1", impl="ours", method="pos", roi="skin", hr_pred=60.0, hr_gt=60.0)])
    tb = pd.DataFrame([
        _row(dataset="MMPD", id="subject5_L4_MO1_E1_S5_GE2_GL2_H1_MA2", subject="5", impl="toolbox", method="pos",
             roi="toolbox_face_box", hr_pred=74.0, hr_gt=70.0, **lab(tb_labels)),
        _row(id="subject1", subject="1", impl="toolbox", method="pos", roi="toolbox_face_box", hr_pred=61.0, hr_gt=60.0),
    ])
    ours.to_csv(tmp_path / "ours.csv", index=False)
    tb.to_csv(tmp_path / "tb.csv", index=False)
    df = load([tmp_path / "ours.csv", tmp_path / "tb.csv"])
    cc = cross_check(df).set_index(["dataset", "ours_roi"])
    assert cc.loc[("MMPD", "face_box"), "n"] == 1
    assert cc.loc[("MMPD", "face_box"), "median_abs_hr_diff"] == pytest.approx(1.0)
    assert cc.loc[("UBFC-rPPG", "skin"), "median_abs_hr_diff"] == pytest.approx(1.0)
    abl = roi_ablation(df)
    assert set(abl[abl["dataset"] == "MMPD"]["roi"]) == {"skin", "face_box"}
