"""ShotMemory / ShotTracker in isolation: group-id assignment, the recall+score
re-identification path, per-shot metadata, and the ABAB recurrence contract. Driven
with synthetic pHashes so the logic is testable without the vision pipeline."""

from types import SimpleNamespace

import numpy as np

import time

from framegate.config import GateConfig
from framegate.reid import _unpack
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
    m = ShotMemory(reid_maxd=0.25, reid_llr=8.0)
    gid, is_new = m.open_shot(rint())
    assert (gid, is_new) == (0, True)
    assert len(m) == 1


def test_distinct_hashes_make_new_groups():
    m = ShotMemory(reid_maxd=0.25, reid_llr=8.0)
    a, b = rint(), rint()  # two random hashes are ~32 bits apart, well outside maxd
    g0, n0 = m.open_shot(a)
    g1, n1 = m.open_shot(b)
    assert (g0, n0) == (0, True)
    assert (g1, n1) == (1, True)


def test_near_duplicate_reids_the_same_group():
    m = ShotMemory(reid_maxd=0.25, reid_llr=8.0)
    base = rint()
    gid, _ = m.open_shot(base)
    for _ in range(30):  # build a confident profile for the group
        m.accumulate(gid, flip(base, 1))
    m.finalize_group(gid)
    again, is_new = m.open_shot(flip(base, 3))  # a new shot, 3 bits off
    assert again == gid and is_new is False


def test_beyond_radius_is_never_matched():
    """A candidate outside reid_maxd is not even retrieved, so it cannot re-ID however
    the scorer would rank it."""
    m = ShotMemory(reid_maxd=0.1, reid_llr=-1e6)  # ~6-bit radius, permissive score
    base = rint()
    gid, _ = m.open_shot(base)
    m.finalize_group(gid)
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
    cfg = GateConfig(reid_maxd=0.25, reid_llr=8.0)
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
    cfg = GateConfig(reid_maxd=0.25, reid_llr=8.0)
    t = ShotTracker(cfg)
    A, B = rint(), rint()
    _run(t, [flip(A, 1)] * 5 + [flip(B, 1)] * 5, cut_before={6})
    shots = t.shots
    assert len(shots) == 1  # only the closed shot is recorded; the last is still open
    s = shots[0]
    assert isinstance(s, Shot)
    assert (s.shot_id, s.group_id) == (0, 0)
    assert s.n_frames == 5  # the 5 A frames; the cut frame belongs to the next shot
    assert s.scene_graph is None  # reserved slot, unused for now


def test_group_accumulates_across_recurrences():
    cfg = GateConfig(reid_maxd=0.25, reid_llr=8.0)
    t = ShotTracker(cfg)
    A, B = rint(), rint()
    out = _run(
        t,
        [flip(A, 1)] * 5 + [flip(B, 1)] * 5 + [flip(A, 2)] * 5,
        cut_before={6, 11},
    )
    # the third block re-identifies to group 0 (the first A), so its frames fold into
    # that group's prototype(s) -- the group's total count exceeds any single shot's.
    assert out[-1][1] == 0  # last frame's group is 0, i.e. A recurred
    assert isinstance(t._mem.group(0), Group)
    t._mem.finalize_group(0)
    total = sum(pr.profile.n for pr in t._mem.prototypes(0))
    assert total >= 9  # first A (5) + the recurrence, less the still-pending frame


# --- multi-prototype drift (a shot that pans/zooms) ------------------------------


def _monotonic_drift(m, gid, base, n=200, order=None):
    """Feed a shot that drifts monotonically from `base` (a pan), one new bit every few
    frames. Returns the final hash and the bit order used."""
    order = order if order is not None else [int(b) for b in rng.permutation(64)]
    h = base
    for i in range(n):
        if i and i % 4 == 0:
            h ^= 1 << order[(i // 4) % 64]
        m.accumulate(gid, h)
    return h, order


def test_small_jitter_stays_one_prototype():
    """The common case -- a near-static shot with a few bits of frame-to-frame jitter
    (a talking head) -- must not spawn extra prototypes."""
    m = ShotMemory(reid_maxd=0.25, reid_llr=8.0)
    base = rint()
    gid, _ = m.open_shot(base)
    for _ in range(150):
        m.accumulate(gid, flip(base, 2))
    m.finalize_group(gid)
    assert len(m.prototypes(gid)) == 1


def test_drift_spawns_prototypes_and_reids_both_ends():
    """A shot that drifts far spawns local prototypes, and a later recurrence near
    either end -- the start or the drifted end -- re-identifies to the same group,
    which a single shot-wide profile could not do."""
    m = ShotMemory(reid_maxd=0.25, reid_llr=8.0)
    base = rint()
    gid, _ = m.open_shot(base)
    end, order = _monotonic_drift(m, gid, base)
    m.finalize_group(gid)
    assert len(m.prototypes(gid)) > 1  # the drift was covered by several prototypes

    g_start, new_start = m.open_shot(base ^ (1 << order[0]))
    g_end, new_end = m.open_shot(end ^ (1 << order[1]))
    assert (g_start, new_start) == (gid, False)
    assert (g_end, new_end) == (gid, False)


def test_distinct_shots_still_separate_with_prototypes():
    """Prototypes must not over-merge: a genuinely different setup is still a new group,
    even after the first shot has spread into several prototypes."""
    m = ShotMemory(reid_maxd=0.25, reid_llr=8.0)
    a = rint()
    ga, _ = m.open_shot(a)
    _monotonic_drift(m, ga, a)
    m.finalize_group(ga)
    b = rint()  # unrelated, ~32 bits from a
    gb, new_b = m.open_shot(b)
    assert new_b is True and gb != ga


def test_accumulate_is_nanoseconds_per_frame(capsys):
    """The per-frame drift check is the whole cost accumulate adds. It must stay a
    single popcount against the active prototype in the common (near) case."""
    m = ShotMemory(reid_maxd=0.25, reid_llr=8.0)
    base = rint()
    gid, _ = m.open_shot(base)
    hs = [flip(base, 2) for _ in range(100_000)]  # near frames -> fast path every time
    t = time.perf_counter()
    for h in hs:
        m.accumulate(gid, h)
    ns = (time.perf_counter() - t) / len(hs) * 1e9
    with capsys.disabled():
        print(f"\n  shotmem accumulate per frame: {ns:.0f} ns")
    assert len(m.prototypes(gid)) == 1  # jitter stayed in one prototype
    assert ns < 3000  # generous for CI; the check itself is ~100 ns


def test_hot_cache_survives_buffer_fold():
    """The hot path caches the active prototype's buffer and its bound append; a fold
    that fires mid-shot (buffer past its bound) must clear that buffer in place, not
    replace it, or the cache would append into a discarded list and silently drop
    frames."""
    m = ShotMemory(reid_maxd=0.25, reid_llr=8.0)
    base = rint()
    gid, _ = m.open_shot(base)
    for _ in range(9000):  # > the 4096 fold bound, so a fold fires during accumulate
        m.accumulate(gid, flip(base, 2))
    m.finalize_group(gid)
    _, _, buf, _ = m._hot[gid]
    assert buf is m.prototypes(gid)[0].profile._buf  # same object across folds
    assert sum(pr.profile.n for pr in m.prototypes(gid)) == 9000  # nothing dropped


# --- deferred accumulation (cut is confirmed one frame late) ---------------------


def test_cut_frame_is_not_folded_into_the_outgoing_shot():
    """The frame that opens a new shot must not contaminate the previous group's
    profile -- it is the one frame guaranteed to show a different setup."""
    cfg = GateConfig(reid_maxd=0.25, reid_llr=8.0)
    t = ShotTracker(cfg)
    A, B = rint(), rint()
    seq = [flip(A, 1) for _ in range(5)] + [flip(B, 1) for _ in range(5)]
    _run(t, seq, cut_before={6})
    t._mem.finalize_group(0)
    counts = sum(pr.profile.counts for pr in t._mem.prototypes(0))
    assert np.array_equal(counts, _unpack(np.array(seq[:5], np.uint64)).sum(0))
    assert t.shots[0].end_frame == 4  # last frame of the A block, not the cut frame


def test_rematch_makes_the_matched_prototype_active():
    """After a re-ID the group's hot prototype must be the one that won the match, not
    whichever it happened to drift onto at the end of its previous occurrence."""
    m = ShotMemory(reid_maxd=0.25, reid_llr=8.0)
    base = rint()
    gid, _ = m.open_shot(base)
    _monotonic_drift(m, gid, base)
    m.finalize_group(gid)
    assert m._hot[gid][0] != base  # drifted off the first prototype
    again, is_new = m.open_shot(base)
    assert (again, is_new) == (gid, False)
    assert m._hot[gid][0] == base  # resumed on the prototype that matched
