"""The front gate: ``DuplicateDetector`` decides whether a frame is byte-identical to the
last one before anything is computed on it. Its answer must be exactly ``np.array_equal``'s,
and a frame that differs -- nearly every frame of real video -- must be rejected in a few
microseconds without a full comparison, or the gate costs more than the duplicates it saves.

These tests pin the predicate, the cheap rejection, the adaptive probe, the handling of
shape and layout changes, and the latency, so that none of it is lost in a later change.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from framegate import Gate, GateConfig
from framegate import gate as gate_module
from framegate.gate import DuplicateDetector

H, W = 720, 1280


def _frame(seed: int, shape=(H, W, 3), dtype=np.uint8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.random(shape).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, int(info.max) + 1, shape, dtype=dtype)


def _flip(a: np.ndarray, index) -> np.ndarray:
    b = a.copy()
    b[index] ^= 1
    return b


class _Counting:
    """Counts the full comparisons the detector makes (``np.array_equal`` in its module)."""

    def __init__(self, monkeypatch):
        self.calls = 0
        real = gate_module.np.array_equal

        def counted(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(gate_module.np, "array_equal", counted)


# --- the predicate -----------------------------------------------------------------------


def test_duplicate_and_distinct_frames():
    d = DuplicateDetector()
    a = _frame(0)
    assert d.same(a, a.copy())
    assert not d.same(a, _frame(1))
    assert d.same(a, a.copy())  # state from the rejection does not leak into the answer


def test_a_single_byte_anywhere_is_a_difference():
    """Flips in rows no probe touches too: the full comparison catches them."""
    d = DuplicateDetector()
    a = _frame(2)
    rng = np.random.default_rng(3)
    for _ in range(300):
        idx = (int(rng.integers(H)), int(rng.integers(W)), int(rng.integers(3)))
        b = _flip(a, idx)
        assert not d.same(a, b), idx
        assert d.same(a, b) == np.array_equal(a, b)
    for row in (
        0,
        1,
        7,
        H // 2 - 1,
        H // 2 + 1,
        H - 2,
        H - 1,
    ):  # around and between the probes
        assert not d.same(a, _flip(a, (row, W - 1, 2)))


def test_the_answer_is_array_equal_whatever_the_state():
    """Random pairs -- duplicates, single-byte, single-row, region and whole-frame changes --
    in random order, so the hot row moves around: the answer never depends on it."""
    d = DuplicateDetector()
    rng = np.random.default_rng(4)
    base = _frame(5, (90, 160, 3))
    kinds = ("dup", "byte", "row", "region", "all")
    for _ in range(400):
        kind = kinds[rng.integers(len(kinds))]
        b = base.copy()
        if kind == "byte":
            b[rng.integers(90), rng.integers(160), rng.integers(3)] ^= 1
        elif kind == "row":
            b[rng.integers(90)] ^= 1
        elif kind == "region":
            r, c = rng.integers(80), rng.integers(150)
            b[r : r + 10, c : c + 10] ^= 1
        elif kind == "all":
            b = _frame(int(rng.integers(1 << 30)), (90, 160, 3))
        assert d.same(base, b) == np.array_equal(base, b), kind
        assert d.same(b, base) == np.array_equal(b, base), kind


def test_never_a_duplicate_when_array_equal_says_no():
    """The direction that matters: a frame is never skipped unless it is identical. Floats
    with NaN are unequal to themselves under array_equal, so they are not duplicates here
    either; a byte difference that array_equal forgives (+0.0 vs -0.0) only rejects."""
    d = DuplicateDetector()
    f = _frame(6, (16, 32, 3), np.float32)
    assert d.same(f, f.copy())
    g = f.copy()
    g[8, 0, 0] = np.nan
    assert not d.same(g, g.copy()) and not np.array_equal(g, g.copy())
    z = np.zeros((16, 32), np.float32)
    nz = z.copy()
    nz[8, 1] = -0.0
    # conservative: processed, not skipped
    assert np.array_equal(z, nz) and not d.same(z, nz)


def test_dtypes_including_object_arrays():
    d = DuplicateDetector()
    for dtype in (
        np.uint8,
        np.int8,
        np.uint16,
        np.int32,
        np.float32,
        np.float64,
        np.bool_,
    ):
        a = (
            _frame(7, (24, 40, 3), dtype)
            if dtype is not np.bool_
            else _frame(7, (24, 40, 3)) > 127
        )
        assert d.same(a, a.copy()), dtype
        b = a.copy()
        b[23, 39, 2] = not b[23, 39, 2] if dtype is np.bool_ else b[23, 39, 2] + 1
        assert not d.same(a, b), dtype
    o = np.array([[1, 2], [3, 4]], dtype=object)
    # equal values, other objects
    assert d.same(o, np.array([[1, 2], [3, 4]], dtype=object))
    assert not d.same(o, np.array([[1, 2], [3, 5]], dtype=object))


def test_different_shapes_and_dtypes_are_never_duplicates():
    d = DuplicateDetector()
    a = _frame(8)
    assert not d.same(a, a[:-1])  # height
    assert not d.same(a, a[:, :-1])  # width
    assert not d.same(a, a[:, :, :2])  # channels
    assert not d.same(a, a.reshape(H * 3, W))  # same bytes, other shape
    assert not d.same(a, a.astype(np.int16))  # dtype
    signed = a.astype(np.uint16).astype(np.uint8).astype(np.int8)
    assert not d.same(a, signed)  # signedness
    plane = a[:, :, 0]
    assert not d.same(plane, plane.reshape(H, W, 1))  # 2-D vs 3-D of one channel


def test_frame_shape_changes_between_calls():
    """A stream whose frames change size: each size is fitted when first seen, a stale hot row
    from a taller frame never indexes a shorter one, and a size seen before is fitted again.
    """
    d = DuplicateDetector()
    sizes = [
        (720, 1280, 3),
        (1080, 1920, 3),
        (360, 640, 3),
        (720, 1280, 3),
        (90, 160),
        (1, 4, 3),
        (4, 4, 4),
    ]
    for i, shape in enumerate(sizes):
        a = _frame(100 + i, shape)
        b = a.copy()
        b[-1] ^= 1  # the last row: the hot row becomes the tallest index of this size
        assert not d.same(a, b), shape
        assert d._hot == shape[0] - 1
        assert d.same(a, a.copy()), shape
        assert not d.same(a, _frame(200 + i, shape)), shape
    # the hot row of a tall frame, then a short frame straight away
    tall, short = _frame(9, (1080, 1920, 3)), _frame(10, (8, 16, 3))
    assert not d.same(tall, _flip(tall, (1079, 0, 0)))
    assert d._hot == 1079
    assert d.same(short, short.copy())
    assert not d.same(short, _flip(short, (7, 15, 2)))


def test_layouts_views_and_strides():
    """Logical content decides, as with array_equal: Fortran order, cropped views, reversed
    views and byte-identical but differently strided copies."""
    d = DuplicateDetector()
    a = _frame(11, (120, 160, 3))
    fa = np.asfortranarray(a)
    assert fa.strides != a.strides
    assert d.same(a, fa) and d.same(fa, a) and d.same(fa, fa.copy())
    assert not d.same(fa, np.asfortranarray(_flip(a, (60, 80, 1))))
    cropped = a[8:-8, 16:-16]  # non-contiguous
    assert d.same(cropped, cropped.copy()) and d.same(cropped.copy(), cropped)
    assert not d.same(cropped, _flip(a, (60, 80, 0))[8:-8, 16:-16])
    rev = a[::-1]  # negative stride
    assert d.same(rev, rev.copy()) and d.same(rev.copy(), rev)
    assert not d.same(rev, a)
    assert d.same(a[::2, ::2], np.ascontiguousarray(a[::2, ::2]))
    assert not d.same(a[::2, ::2], a[1::2, ::2])


def test_small_and_degenerate_frames():
    d = DuplicateDetector()
    for n in (1, 2, 3, 4, 5, 6, 7, 8):  # probe rows coincide for the tiny ones
        a = _frame(20 + n, (n, 6, 3))
        assert d.same(a, a.copy()), n
        for r in range(n):
            assert not d.same(a, _flip(a, (r, 5, 1))), (n, r)
    one_d = _frame(30, (64,))
    assert d.same(one_d, one_d.copy()) and not d.same(one_d, _flip(one_d, 63))
    scalar = np.array(3, np.uint8)
    assert d.same(scalar, scalar.copy()) and not d.same(scalar, np.array(4, np.uint8))
    for shape in ((0, 3), (0, 16, 3), (16, 0, 3), (16, 16, 0)):
        e = np.zeros(shape, np.uint8)
        assert d.same(e, e.copy()), shape
        assert not d.same(e, np.zeros((*shape[:-1], shape[-1] + 1), np.uint8)), shape


def test_one_buffer_under_two_names_is_not_a_duplicate():
    """A caller that decodes into a reused buffer hands the gate the same array twice; the
    previous frame's bytes are gone, so there is nothing to reuse and the frame is processed.
    """
    d = DuplicateDetector()
    a = _frame(12)
    assert not d.same(a, a)
    view = a[:]
    assert d.same(a, view)  # a distinct view object of the same bytes is a duplicate
    assert np.array_equal(a, a)  # (what the old predicate answered)


# --- the cheap rejection -------------------------------------------------------------------


def test_a_changed_frame_is_rejected_without_a_full_comparison(monkeypatch):
    """The point of the probes: the block scan runs only for duplicates and for frames that
    differ in no probed row, and a duplicate is the only frame compared in full (16 blocks).
    """
    counting = _Counting(monkeypatch)
    d = DuplicateDetector()
    a = _frame(13)
    assert not d.same(a, _frame(14))  # differs everywhere: one probe
    assert not d.same(a, _flip(a, (H // 2, 0, 0)))  # at the halfway row
    assert not d.same(a, _flip(a, (H - 1, 0, 0)))  # at the last row: a later probe
    assert counting.calls == 0
    assert d.same(a, a.copy())
    assert counting.calls == 16  # every block, once
    counting.calls = 0
    # no probe looks here: the scan stops at the block that differs (the 9th of 16)
    assert not d.same(a, _flip(a, (H // 2 + 3, 0, 0)))
    assert 0 < counting.calls <= 9
    counting.calls = 0
    # ... and taught the probe its row
    assert not d.same(a, _flip(a, (H // 2 + 3, 0, 0)))
    assert counting.calls == 0


def test_the_scan_teaches_the_probe_the_row_that_changed():
    """A change that sits where no fixed probe looks: one scan finds its first row, then that
    row rejects every following frame of the band by itself."""
    d = DuplicateDetector()
    a = _frame(40)
    d.same(a, a.copy())
    for top in (3, 200, 371, 600, H - 7):
        b = a.copy()
        b[top : top + 5] ^= 1
        assert not d.same(a, b)
        assert d._hot == top, top  # the first differing row of the band
        assert not d.same(a, b)
        assert d._hot == top
    # a differing block whose rows all look the same as bytes (NaN against NaN) is still a
    # difference under array_equal: rejected, with the probe left where it was
    f = _frame(41, (64, 8), np.float32)
    d.same(f, f.copy())
    hot = d._hot
    g = f.copy()
    g[50, 2] = np.nan
    assert not d.same(g, g.copy()) and d._hot == hot


def test_scan_blocks_cover_the_frame_for_every_height():
    d = DuplicateDetector()
    for n in (1, 2, 3, 15, 16, 17, 100, H, 1080):
        a = _frame(50 + n, (n, 4))
        d.same(a, a.copy())
        blocks = d._blocks
        assert len(blocks) == min(16, n)
        assert blocks[0][0] == 0 and blocks[-1][1] == n
        assert all(r0 < r1 for r0, r1 in blocks)
        assert all(blocks[i][1] == blocks[i + 1][0] for i in range(len(blocks) - 1))


def test_the_probe_that_rejected_last_time_goes_first(monkeypatch):
    """A change confined to one band: after one miss the hot row is in the band and every
    following frame is rejected by the first probe, without a full comparison."""
    counting = _Counting(monkeypatch)
    d = DuplicateDetector()
    a = _frame(15)
    d.same(a, a.copy())  # fits the probes: hot = the halfway row
    assert d._hot == H // 2
    b = a.copy()
    b[H - 1] ^= 1  # only the last row changes
    assert not d.same(a, b)
    assert d._hot == H - 1
    for _ in range(5):
        assert not d.same(a, b)
    assert d._hot == H - 1
    c = a.copy()
    c[H // 4] ^= 1
    assert not d.same(a, c)
    assert d._hot == H // 4
    assert (
        counting.calls == 16
    )  # the one duplicate at the start, scanned block by block


def test_the_probe_rows_are_fixed_per_shape():
    d = DuplicateDetector()
    a = _frame(16)
    d.same(a, a.copy())
    assert d._rows == (H // 2, H // 4, (3 * H) // 4, H - 1)
    rows = d._rows
    for _ in range(3):
        d.same(a, _frame(17))
        assert d._rows is rows  # not rebuilt while the shape holds
    d.same(_frame(18, (360, 640, 3)), _frame(19, (360, 640, 3)))
    assert d._rows == (180, 90, 270, 359)


# --- in the gate ---------------------------------------------------------------------------


def test_gate_reuses_stats_only_for_byte_identical_frames():
    g = Gate(GateConfig(skip_duplicates=True))
    assert isinstance(g._dup, DuplicateDetector)
    a = _frame(21, (180, 320, 3))
    fs1, _ = g.frame(a)
    fs2, _ = g.frame(a.copy())
    assert fs2 is fs1
    fs3, _ = g.frame(_flip(a, (100, 100, 0)))
    assert fs3 is not fs1
    fs4, _ = g.frame(a)  # the frame from two calls ago is not the last one
    assert fs4 is not fs1 and fs4 is not fs3
    g.close()


def test_gate_with_skip_duplicates_off_processes_every_frame():
    g = Gate(GateConfig(skip_duplicates=False))
    a = _frame(22, (180, 320, 3))
    fs1, _ = g.frame(a)
    fs2, _ = g.frame(a.copy())
    assert fs2 is not fs1
    g.close()


def test_gates_do_not_share_detector_state():
    g1, g2 = Gate(GateConfig()), Gate(GateConfig())
    a = _frame(23, (180, 320, 3))
    g1.frame(a)
    g1.frame(_flip(a, (179, 0, 0)))
    assert g1._dup._hot == 179
    g2.frame(a)
    g2.frame(a.copy())
    assert g2._dup._hot == 90
    g1.close()
    g2.close()


def test_gate_handles_a_change_of_frame_size_mid_stream():
    g = Gate(GateConfig())
    a = _frame(24, (180, 320, 3))
    g.frame(a)
    b = _frame(25, (360, 640, 3))
    fsb, _ = g.frame(b)
    fsb2, _ = g.frame(b.copy())
    assert fsb2 is fsb
    fsa, _ = g.frame(a)
    assert fsa is not fsb
    g.close()


# --- its latency ---------------------------------------------------------------------------


def _median_us(fn, n: int) -> float:
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return 1e6 * ts[n // 2]


@pytest.mark.parametrize(
    ("label", "b_of"),
    [
        ("frames that differ everywhere", lambda a: _frame(31)),
        (
            "a change in one band (last rows)",
            lambda a: _flip(a, (slice(H - 20, H), slice(None), 0)),
        ),
        ("a change in no fixed probe row", lambda a: _flip(a, (H // 2 + 3, 0, 0))),
    ],
)
def test_latency_of_rejecting_a_frame(label, b_of):
    """A frame that differs is rejected in microseconds -- after one scan has taught the probe
    where the change is, when it sits where no fixed probe looks."""
    d = DuplicateDetector()
    a = _frame(30)
    b = b_of(a)
    d.same(a, b)  # fit, and set the hot row (for the last case: by the scan)
    us = _median_us(lambda: d.same(a, b), 2000)
    print(f"  front gate, {label}: {us:.2f} us per frame")
    assert us < 25.0, f"{label}: {us:.1f} us"


def test_latency_of_the_scan_and_of_a_duplicate():
    """The cold path: a change the probes missed costs one scan that stops at its block; a
    duplicate is compared in full. Both are the price of one such frame, not of every frame.
    """
    d = DuplicateDetector()
    a = _frame(32)
    c = a.copy()
    assert d.same(a, c)
    us = _median_us(lambda: d.same(a, c), 200)
    print(f"  front gate, a duplicate (every block compared): {us:.0f} us per frame")
    assert us < 5000.0
    b = _flip(a, (H // 2 + 3, 0, 0))

    def miss():
        d._hot = d._rows[
            0
        ]  # forget what the scan learned, so each call is a first miss
        d.same(a, b)

    us = _median_us(miss, 200)
    print(
        f"  front gate, first miss of a change in no fixed probe row (scan): {us:.0f} us"
    )
    assert us < 5000.0
