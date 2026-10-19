import numpy as np
import pytest

from afscreen.face import patches
from afscreen.face.patches_def import FINE_PATCHES, GROUPS


def test_levels_partition_the_fine_patches():
    assert len(FINE_PATCHES) == 25
    for level, members in GROUPS.items():
        assert len(members) == level
        assert sorted(i for m in members for i in m) == list(range(25))
    assert patches.LEVELS == (25, 12, 6, 1)


def _fake_points(seed=0):
    rng = np.random.default_rng(seed)
    pts = np.zeros((478, 3))
    pts[:, :2] = rng.uniform(20, 180, (478, 2))
    return pts


def test_measure_uniform_frame_and_flat_face():
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[:] = (200, 120, 90)
    rgb, counts, facing = patches.measure(frame, _fake_points())
    seen = counts > 0
    assert seen.sum() > 15
    assert np.allclose(rgb[seen], [200, 120, 90])
    assert np.allclose(facing, 1.0)  # z = 0 everywhere: every patch faces the camera


def test_merge_is_pixel_weighted():
    t = 4
    rgb = np.zeros((t, 25, 3))
    rgb[:, 0] = 100.0
    rgb[:, 4] = 200.0
    counts = np.zeros((t, 25))
    counts[:, 0], counts[:, 4] = 300, 100
    facing = np.ones((t, 25))
    m_rgb, m_n, _ = patches.merge(rgb, counts, facing, 12)
    j = next(j for j, m in enumerate(GROUPS[12]) if 0 in m)
    assert 4 in GROUPS[12][j]
    assert np.allclose(m_rgb[:, j], 125.0)
    assert np.allclose(m_n[:, j], 400)
    one, _, _ = patches.merge(rgb, counts, facing, 1)
    assert one.shape == (t, 1, 3)


@pytest.mark.parametrize("px, level", [(800, 25), (250, 12), (90, 6), (5, 6)])
def test_choose_level_by_face_size(px, level):
    counts = np.full((50, 25), float(px))
    assert patches.choose_level(counts, min_pixels=350) == level
