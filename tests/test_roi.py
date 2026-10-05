import numpy as np

from rppg.face.roi import ROI_POLYGONS, face_box, polygon_mask, roi_masks


def test_polygon_mask_area():
    mask = polygon_mask((100, 100), np.array([[10, 10], [49, 10], [49, 29], [10, 29]], float))
    assert mask.sum() == 40 * 20


def fake_landmarks():
    """478 landmarks where each ROI polygon is a distinct 10x10 square."""
    pts = np.full((478, 2), 50.0)
    offsets = {"forehead": (40, 10), "right_cheek": (20, 40), "left_cheek": (60, 40)}
    for name, (x, y) in offsets.items():
        idx = ROI_POLYGONS[name]
        corners = np.array([[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]], float)
        pts[idx] = corners[np.arange(len(idx)) * 4 // len(idx)]
    return pts


def test_roi_masks_are_disjoint_and_nonempty():
    masks = roi_masks((100, 100), fake_landmarks())
    assert set(masks) == set(ROI_POLYGONS)
    assert all(m.sum() > 50 for m in masks.values())
    total = sum(m.astype(int) for m in masks.values())
    assert total.max() == 1


def test_face_box_is_enlarged_and_clipped():
    pts = np.array([[40, 40], [60, 60]], float)
    assert face_box((100, 100), pts, scale=1.5) == (35, 35, 65, 65)
    assert face_box((100, 100), pts, scale=10) == (0, 0, 100, 100)
