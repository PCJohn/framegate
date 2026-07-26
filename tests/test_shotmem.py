"""ShotMemory / ShotTracker in isolation: group-id assignment, the recall+score
re-identification path, per-shot metadata, and the ABAB recurrence contract. Driven
with synthetic pHashes so the logic is testable without the vision pipeline."""

from types import SimpleNamespace

import numpy as np

from framegate.config import GateConfig
from framegate.shotmem import Group, Shot, ShotMemory, ShotTracker

rng = np.random.default_rng(0)


def rint():
    return int(rng.integers(0, 2**64, dtype=np.uint64))


def flip(h, nbits):
    """`h` with `nbits` random bits flipped -- a near-duplicate at that Hamming dist."""
    for b in rng.choice(64, nbits, replace=False):
        h ^= 1 << int(b)
    return h


# --- ShotMemory: the recall + score core -----------------------------------------


def test_first_shot_opens_group_zero():
    m = ShotMemory(reid_maxd=0.25, reid_ll=-0.28)
    gid, is_new = m.open_shot(rint())
    assert (gid, is_new) == (0, True)
    assert len(m) == 1


def test_distinct_hashes_make_new_groups():
    m = ShotMemory(reid_maxd=0.25, reid_ll=-0.28)
    a, b = rint(), rint()  # two random hashes are ~32 bits apart, well outside maxd
    g0, n0 = m.open_shot(a)
    g1, n1 = m.open_shot(b)
    assert (g0, n0) == (0, True)
    assert (g1, n1) == (1, True)


def test_near_duplicate_reids_the_same_group():
    m = ShotMemory(reid_maxd=0.25, reid_ll=-0.28)
    base = rint()
    gid, _ = m.open_shot(base)
    for _ in range(30):  # build a confident profile for the group
        m.accumulate(gid, flip(base, 1))
    m.profile(gid).finalize()
    again, is_new = m.open_shot(flip(base, 3))  # a new shot, 3 bits off
    assert again == gid and is_new is False


def test_beyond_radius_is_never_matched():
    """A candidate outside reid_maxd is not even retrieved, so it cannot re-ID however
    the scorer would rank it."""
    m = ShotMemory(reid_maxd=0.1, reid_ll=-1.0)  # ~6-bit radius, permissive score
    base = rint()
    gid, _ = m.open_shot(base)
    m.profile(gid).finalize()
    far, is_new = m.open_shot(flip(base, 20))  # 20 bits > radius
    assert far != gid and is_new is True


# --- ShotTracker: cut timing, metadata, recurrence -------------------------------


def _sig(cut=False):
    return SimpleNamespace(cut=cut, freeze=False)


def _stat(h):
    return SimpleNamespace(phash=h, blank=False)


def _run(tracker, hashes, cut_before):
    """Feed a hash sequence; `cut_before` is the set of indices that open a new shot."""
    out = []
    for i, h in enumerate(hashes):
        out.append(tracker.update(_stat(h), _sig(cut=i in cut_before), frame_id=i))
    return out


def test_tracker_counts_shots_and_reids_abab():
    cfg = GateConfig(reid_maxd=0.25, reid_ll=-0.28)
    t = ShotTracker(cfg)
    A, B = rint(), rint()
    # blocks of 5 frames; cut is confirmed one frame late, so it fires on the SECOND
    # frame of each new block (indices 6, 11, 16).
    seq = (
        [flip(A, 1) for _ in range(5)]
        + [flip(B, 1) for _ in range(5)]
        + [flip(A, 2) for _ in range(5)]
        + [flip(B, 2) for _ in range(5)]
    )
    out = _run(t, seq, cut_before={6, 11, 16})
    group_ids = [g for _, g in out]
    assert group_ids == [0] * 6 + [1] * 5 + [0] * 5 + [1] * 4  # ABAB -> 2 groups
    assert t.n_groups == 2


def test_tracker_records_shot_metadata():
    cfg = GateConfig(reid_maxd=0.25, reid_ll=-0.28)
    t = ShotTracker(cfg)
    A, B = rint(), rint()
    _run(t, [flip(A, 1)] * 5 + [flip(B, 1)] * 5, cut_before={6})
    shots = t.shots
    assert len(shots) == 1  # only the closed shot is recorded; the last is still open
    s = shots[0]
    assert isinstance(s, Shot)
    assert (s.shot_id, s.group_id) == (0, 0)
    assert s.n_frames == 6  # frames fed before the one-late cut closed the shot
    assert s.scene_graph is None  # reserved slot, unused for now


def test_group_accumulates_across_recurrences():
    cfg = GateConfig(reid_maxd=0.25, reid_ll=-0.28)
    t = ShotTracker(cfg)
    A, B = rint(), rint()
    out = _run(
        t,
        [flip(A, 1)] * 5 + [flip(B, 1)] * 5 + [flip(A, 2)] * 5,
        cut_before={6, 11},
    )
    # the third block re-identifies to group 0 (the first A), so its frames fold into
    # the same profile -- the group's cumulative count exceeds any single shot's length.
    assert out[-1][1] == 0  # last frame's group is 0, i.e. A recurred
    assert isinstance(t._mem.group(0), Group)
    assert t._mem.profile(0).finalize().n >= 10  # first A (6) + recurrence, one profile
