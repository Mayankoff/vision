import json

import cv2
import numpy as np

from afscreen import synth
from afscreen.eval.benchmark import run, score_video
from afscreen.eval.metrics import ibi_error_ms, summarize
from afscreen.io.datasets import load_pure, load_ubfc, load_ubfc1
from afscreen.traces import PatchTraces


def _write_video(path, n=10, size=(32, 24)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    for _ in range(n):
        w.write(np.full((size[1], size[0], 3), 128, np.uint8))
    w.release()


def test_load_ubfc(tmp_path):
    d = tmp_path / "subject3"
    d.mkdir()
    _write_video(d / "vid.avi")
    t = np.arange(10) / 30
    (d / "ground_truth.txt").write_text(" ".join(map(str, np.sin(t))) + "\n" + "70 " * 10 + "\n" + " ".join(map(str, t)))
    (rec,) = load_ubfc(str(tmp_path))
    assert rec.id == "subject3" and rec.subject == "3"
    assert len(list(rec.frames())) == 10
    assert len(rec.aligned_bvp(10)) == 10


def test_load_ubfc1(tmp_path):
    d = tmp_path / "8-gt"
    d.mkdir()
    _write_video(d / "vid.avi")
    rows = [f"{1000 + 16 * i},70,97,{i % 7}" for i in range(20)]
    (d / "gtdump.xmp").write_text("\n".join(rows))
    (rec,) = load_ubfc1(str(tmp_path))
    assert rec.id == "ubfc1_8-gt" and len(rec.bvp) == 20
    assert len(rec.aligned_bvp(10)) == 10


def test_load_pure(tmp_path):
    d = tmp_path / "01-02"
    (d / "01-02").mkdir(parents=True)
    ts = [1_000_000_000 + i * 33_333_333 for i in range(5)]
    for t in ts:
        cv2.imwrite(str(d / "01-02" / f"Image{t}.png"), np.zeros((8, 8, 3), np.uint8))
    data = {"/FullPackage": [{"Timestamp": ts[0] + i * 16_666_666, "Value": {"waveform": i}} for i in range(10)],
            "/Image": [{"Timestamp": t} for t in ts]}
    (d / "01-02.json").write_text(json.dumps(data))
    (rec,) = load_pure(str(tmp_path))
    assert rec.meta["session"] == "02"
    assert np.allclose(rec.aligned_bvp(5), [0, 2, 4, 6, 8], atol=0.01)


def test_traces_roundtrip(tmp_path):
    tr = PatchTraces(fps=30.0, rgb=np.random.rand(20, 25, 3), counts=np.ones((20, 25)), facing=np.ones((20, 25)),
                     pose=np.zeros((20, 6)), meta={"id": "x"}, ref_pulse=np.arange(20.0), ref_beats=np.array([0.5, 1.2]))
    tr.save(tmp_path / "x.npz")
    back = PatchTraces.load(tmp_path / "x.npz")
    assert back.meta == {"id": "x"} and np.allclose(back.ref_beats, [0.5, 1.2])
    assert np.allclose(back.rgb, tr.rgb, atol=1e-5)


def test_metrics():
    m = summarize([70, 80, 90], [72, 80, 87])
    assert m["n"] == 3 and abs(m["MAE"] - 5 / 3) < 1e-9
    ref = np.arange(1, 20) * 0.8
    assert ibi_error_ms(ref + 0.01, ref) < 1e-6
    assert abs(ibi_error_ms(ref + np.tile([0, 0.02], 10)[:19], ref) - 20) < 1e-6


def test_align_reference_undoes_delay_and_inversion():
    from afscreen.eval.reference import align_reference

    fs = 30.0
    t = np.arange(600) / fs
    pred = synth.pulse_from_beats(t, synth.regular_beats(20, 72, seed=3))
    ref = -np.roll(pred, -5)  # the sensor sees the pulse 5 frames earlier, upside down
    aligned, lag, sign = align_reference(ref, pred, fs)
    assert sign == -1 and abs(lag - 5 / fs) < 1e-9
    assert np.corrcoef(aligned[10:-10], pred[10:-10])[0, 1] > 0.99


def test_benchmark_on_synthetic_traces(tmp_path):
    beats = synth.regular_beats(30, 72, seed=1)
    d = synth.patch_traces(beats, 30.0, 30.0, seed=2)
    n = d["rgb"].shape[0]
    tr = PatchTraces(fps=30.0, rgb=d["rgb"], counts=np.full((n, 25), 500.0), facing=d["facing"], pose=d["pose"],
                     meta={"dataset": "synthetic", "id": "a"}, ref_pulse=d["pulse"], ref_beats=beats)
    rows = score_video(tr, ("single", "full"))
    assert all(r["abs_err"] < 1.0 for r in rows)
    tr.save(tmp_path / "cache" / "a.npz")
    summary = run(tmp_path / "cache", tmp_path / "out", ("single", "full"))
    assert list(summary.variant) == ["single", "full"]
    assert (tmp_path / "out" / "summary.md").exists()
