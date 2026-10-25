import numpy as np
import pytest

from afscreen import synth
from afscreen.pipeline import ALL_VARIANTS, BASELINES, analyse
from afscreen.signal.baselines import chrom, green
from afscreen.signal.filters import clean_pulse
from afscreen.signal.spectrum import hr_fft
from afscreen.traces import PatchTraces


@pytest.mark.parametrize("method", [chrom, green])
@pytest.mark.parametrize("bpm", [45, 72, 150])
def test_baselines_recover_heart_rate(method, bpm):
    d = synth.patch_traces(synth.regular_beats(30, bpm, seed=1), 30.0, 30.0, seed=2)
    assert abs(hr_fft(clean_pulse(method(d["rgb"][:, 0], 30.0), 30.0), 30.0) - bpm) < 2


def test_every_variant_including_baselines_runs():
    d = synth.patch_traces(synth.regular_beats(30, 72, seed=3), 30.0, 30.0, seed=4)
    n = d["rgb"].shape[0]
    tr = PatchTraces(fps=30.0, rgb=d["rgb"], counts=np.full((n, 25), 500.0), facing=d["facing"], pose=d["pose"])
    for v in ALL_VARIANTS:
        r = analyse(tr, v)
        assert abs(r.hr_bpm - 72) < 2, v
        assert (r.level == 1) == (v in BASELINES or v == "single")
