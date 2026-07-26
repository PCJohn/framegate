"""ShotProfile / ShotScorer in isolation: exactness against the naive float model,
the loose-match behaviour the design exists for, and the per-frame / per-cut latency
budget. No vision pipeline, no framestore -- just the scoring core."""

import time

import numpy as np

from framegate.reid import BITS, ShotProfile, ShotScorer, _unpack

rng = np.random.default_rng(0)


def rint(n=None):
    return rng.integers(0, 2**64, n, dtype=np.uint64)


def profile(frames):
    p = ShotProfile(int(frames[0]))
    for f in frames:
        p.add(int(f))
    return p.finalize()


def naive_mean_ll(q, counts, n):
    """The definition, computed the slow obvious way, as an oracle."""
    p = (counts + 0.5) / (n + 1.0)
    qb = _unpack(np.array([q], np.uint64))[0]
    return float(np.where(qb == 1, np.log(p), np.log(1.0 - p)).mean())


def test_scorer_matches_naive_float_model_exactly():
    sc = ShotScorer()
    for _ in range(500):
        n = int(rng.integers(1, 400))
        frames = rint(n)
        p = profile(frames)
        q = int(rint())
        assert abs(sc.score(q, [p])[0] - naive_mean_ll(q, p.counts, p.n)) < 1e-9


def test_counts_are_correct_and_finalize_is_idempotent():
    frames = rint(50)
    p = profile(frames)
    want = _unpack(frames).sum(0)
    assert np.array_equal(p.counts, want)
    assert p.n == 50
    log1_before = p.log1.copy()
    p.finalize()  # again, nothing buffered
    assert p.n == 50 and np.array_equal(p.log1, log1_before)


def test_incremental_finalize_folds_only_new_frames():
    first = rint(30)
    p = profile(first)
    more = rint(20)
    for f in more:
        p.add(int(f))
    p.finalize()
    assert p.n == 50
    assert np.array_equal(p.counts, _unpack(np.concatenate([first, more])).sum(0))


def test_identical_frames_give_confident_bits():
    """A shot that never changes -> every bit is 0 or n, probabilities saturate, and a
    matching query scores near 0 while its complement scores very negative."""
    h = int(rint())
    p = profile(np.full(40, h, np.uint64))
    sc = ShotScorer()
    assert sc.score(h, [p])[0] > -0.05
    assert sc.score(h ^ ((1 << 64) - 1), [p])[0] < -2.0


def test_within_shot_flicker_bits_do_not_penalise_a_rematch():
    """The core design claim. A shot whose 'mouth region' (a few bits) flickers every
    frame must still re-ID a new frame of itself with a new mouth position, because
    those bits sit at p=0.5 and contribute the same -log 2 to any query."""
    base = int(rint())
    flicker = rng.choice(BITS, 8, replace=False)

    def frame():
        m = 0
        for b in flicker:
            if rng.integers(0, 2):
                m |= 1 << int(b)
        return base ^ m

    p = profile([frame() for _ in range(150)])
    sc = ShotScorer()

    same = sc.score(frame(), [p])[0]  # new frame, new mouth
    stable = [b for b in range(BITS) if b not in flicker]
    off = rng.choice(stable, 18, replace=False)
    diff = sc.score(base ^ sum(1 << int(b) for b in off), [p])[0]  # different setup

    assert same > -0.2  # comfortably matches
    assert diff < -1.0  # clearly rejected
    assert same - diff > 0.8  # wide separation to place a threshold in


def test_scorer_ranks_and_batches_consistently():
    """Scoring K profiles at once must equal scoring them one at a time."""
    profs = [profile(rint(int(rng.integers(20, 200)))) for _ in range(8)]
    q = int(rint())
    sc = ShotScorer()
    batched = sc.score(q, profs)
    singly = np.array([sc.score(q, [p])[0] for p in profs])
    assert np.allclose(batched, singly)
    assert sc.score(q, []).shape == (0,)


def test_add_is_nanoseconds_per_frame(capsys):
    """Per-frame cost is the whole point of deferring the fold: it must be a bare
    append, not a hash unpack. Budget generously for CI noise."""
    p = ShotProfile(0)
    hs = [int(x) for x in rint(100_000)]
    t = time.perf_counter()
    for h in hs:
        p.add(h)
    ns = (time.perf_counter() - t) / len(hs) * 1e9
    with capsys.disabled():
        print(f"\n  reid add per frame: {ns:.0f} ns")
    assert ns < 2000


def test_finalize_and_score_are_microseconds(capsys):
    frames = rint(300)
    t = time.perf_counter()
    for _ in range(1000):
        profile(frames)
    fin_us = (time.perf_counter() - t) / 1000 * 1e6

    profs = [profile(rint(200)) for _ in range(8)]
    q = int(rint())
    sc = ShotScorer()
    t = time.perf_counter()
    for _ in range(5000):
        sc.score(q, profs)
    score_us = (time.perf_counter() - t) / 5000 * 1e6
    with capsys.disabled():
        print(f"  reid finalize (300 frames): {fin_us:.1f} us")
        print(f"  reid score 8 candidates:    {score_us:.1f} us")
    assert fin_us < 500 and score_us < 200


def test_periodic_fold_matches_single_fold():
    """A long take folds its buffer periodically to bound memory; the resulting counts
    and scores must be identical to folding once at the end."""
    frames = rint(10000)
    incremental = ShotProfile(int(frames[0]))
    for f in frames:
        incremental.add(int(f))
        if incremental.pending >= 4096:
            incremental.fold()
    incremental.finalize()

    single = profile(frames)
    assert incremental.n == single.n == 10000
    assert np.array_equal(incremental.counts, single.counts)
    assert np.allclose(incremental.log1, single.log1)
