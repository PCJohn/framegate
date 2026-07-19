"""ShotMemory in isolation: id assignment, threshold, drift refresh, and the
retrieval seam -- exercised with a trivial 1-D score so the logic is testable
without the vision pipeline."""

import numpy as np

from framegate.shotmem import LinearIndex, ShotMemory, ShotRef


def ref(x):  # a descriptor whose "distance" is just |color[0] - .|
    return ShotRef(np.zeros((2, 2), np.float32), np.array([x, 0, 0], np.float32))


def dist(q, r):
    return abs(float(q.color[0]) - float(r.color[0]))


def test_first_match_creates_group_zero():
    m = ShotMemory(threshold=5.0)
    gid, is_new = m.match(ref(0.0), dist)
    assert (gid, is_new) == (0, True)
    assert len(m) == 1


def test_far_descriptor_makes_new_group_near_one_reuses():
    m = ShotMemory(threshold=5.0)
    m.match(ref(0.0), dist)  # group 0
    assert m.match(ref(100.0), dist) == (1, True)  # far -> new group
    assert m.match(ref(1.0), dist) == (0, False)  # within 5 of group 0 -> reuse
    assert m.match(ref(101.0), dist) == (1, False)  # within 5 of group 1 -> reuse
    assert len(m) == 2


def test_threshold_is_inclusive_and_picks_nearest():
    m = ShotMemory(threshold=5.0)
    m.match(ref(0.0), dist)  # group 0
    m.match(ref(10.0), dist)  # group 1
    assert m.match(ref(5.0), dist) == (0, False)  # exactly at threshold from group 0
    assert m.match(ref(7.0), dist) == (1, False)  # nearer group 1 (3) than group 0 (7)


def test_refresh_tracks_drift():
    m = ShotMemory(threshold=5.0)
    gid, _ = m.match(ref(0.0), dist)  # group 0 ref at 0
    m.refresh(gid, ref(20.0))  # drift the reference to 20
    assert m.match(ref(22.0), dist) == (0, False)  # now matches near 20
    assert m.match(ref(2.0), dist) == (1, True)  # old location no longer matches


def test_linear_index_returns_all_added():
    idx = LinearIndex()
    idx.add(0, ref(0.0))
    idx.add(1, ref(9.0))
    assert list(idx.query(ref(3.0))) == [0, 1]  # every group is a candidate
