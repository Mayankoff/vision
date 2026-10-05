from make_splits import KNOWN_SUBJECTS, make_split


def test_splits_are_disjoint_complete_and_deterministic():
    for name, subjects in KNOWN_SUBJECTS.items():
        s = make_split(subjects)
        parts = [set(s[k]) for k in ("train", "val", "test")]
        assert set().union(*parts) == set(subjects), name
        assert sum(len(p) for p in parts) == len(subjects), name
        assert all(parts), name
        assert make_split(subjects) == s


def test_ubfc_has_42_subjects():
    assert len(KNOWN_SUBJECTS["ubfc"]) == 42
