import json

import cv2
import numpy as np
import pytest
from scipy.io import savemat

from rppg.io.datasets import load_mmpd, load_pure, load_ubfc


def write_video(path, n, size=(32, 24), fps=30.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), i % 255, np.uint8))
    w.release()


def test_ubfc_loader(tmp_path):
    d = tmp_path / "subject7"
    d.mkdir()
    write_video(d / "vid.avi", 20)
    t = np.arange(20) / 30
    bvp = np.sin(t)
    (d / "ground_truth.txt").write_text("\n".join(" ".join(map(str, row)) for row in (bvp, np.full(20, 70), t)))
    (rec,) = load_ubfc(str(tmp_path))
    assert (rec.id, rec.subject, rec.dataset) == ("subject7", "7", "UBFC-rPPG")
    assert rec.fps == pytest.approx(30)
    frames = list(rec.frames())
    assert len(frames) == 20 and frames[0].shape == (24, 32, 3)
    np.testing.assert_allclose(rec.aligned_bvp(20), bvp)


def test_pure_loader_interpolates_bvp_to_frame_times(tmp_path):
    name = "03-02"
    (tmp_path / name / name).mkdir(parents=True)
    t0 = 1_000_000_000_000
    img_t = [t0 + int(i * 1e9 / 30) for i in range(6)]
    for ts in img_t:
        cv2.imwrite(str(tmp_path / name / name / f"Image{ts}.png"), np.zeros((8, 8, 3), np.uint8))
    bvp_t = [t0 + int(i * 1e9 / 60) for i in range(12)]  # 60 Hz oximeter
    doc = {
        "/FullPackage": [{"Timestamp": ts, "Value": {"waveform": i * 10}} for i, ts in enumerate(bvp_t)],
        "/Image": [{"Timestamp": ts} for ts in img_t],
    }
    (tmp_path / name / f"{name}.json").write_text(json.dumps(doc))
    (rec,) = load_pure(str(tmp_path))
    assert (rec.subject, rec.meta["session"]) == ("03", "02")
    assert len(list(rec.frames())) == 6
    np.testing.assert_allclose(rec.aligned_bvp(6), np.arange(6) * 20, atol=1e-6)  # every 2nd 60 Hz sample


def test_mmpd_loader_reads_labels_and_float_video(tmp_path):
    d = tmp_path / "subject12"
    d.mkdir()
    video = np.random.default_rng(0).random((5, 6, 6, 3)).astype(np.float32)  # mini-MMPD stores floats in [0, 1]
    savemat(
        d / "p12_3.mat",
        {
            "video": video,
            "GT_ppg": np.arange(5.0)[None, :],
            "light": "LED-low",
            "motion": "Walking",
            "exercise": "False",
            "skin_color": np.array([[5]]),
            "gender": "female",
            "glasser": "False",
            "hair_cover": "True",
            "makeup": "False",
        },
    )
    (rec,) = load_mmpd(str(tmp_path))
    assert (rec.id, rec.subject) == ("p12_3", "12")
    assert rec.meta["motion"] == "Walking" and rec.meta["skin_color"] == "5"
    frames = list(rec.frames())
    assert frames[0].dtype == np.uint8 and frames[0].max() > 1
    np.testing.assert_allclose(rec.bvp, np.arange(5.0))


def test_downscale_limits_longer_side():
    from rppg.io.datasets import downscale

    frames = [np.zeros((1080, 1920, 3), np.uint8), np.zeros((300, 200, 3), np.uint8)]
    out = list(downscale(iter(frames), 720))
    assert out[0].shape == (405, 720, 3) and out[1].shape == (300, 200, 3)
