"""The temporal layer's optimizations change no bits.

``StreamAnalyzer.update`` carries each frame's luma statistics to the next frame, keeps the
shift search's working arrays, and holds its histories in rings. Every one of those is a
rearrangement of the same numpy operations on the same values, so every output must equal,
bit for bit, what the straightforward formulation computes. That formulation is kept here
as the reference, and the two are run side by side on real gate output.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from framegate import Gate, GateConfig
from framegate import signals as S
from framegate.stream import (
    StreamAnalyzer,
    TemporalSignals,
    _Luma,
    _RollingRobust,
    _VHistory,
)

# --- the reference: the layer as first written, kept verbatim ------------------------------


def _ref_ncc0(prev, cur):
    a, b = prev - prev.mean(), cur - cur.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def _ref_best_shift(prev, cur, s):
    if s <= 0:
        return _ref_ncc0(prev, cur), 0, 0
    from numpy.lib.stride_tricks import sliding_window_view

    g = prev.shape[-1]
    h = g - 2 * s
    c = cur[s : s + h, s : s + h]
    c = c - c.mean()
    nc = np.sqrt((c * c).sum()) + 1e-6
    wins = sliding_window_view(prev, (h, h)).reshape(-1, h, h)
    a = wins - wins.mean(axis=(1, 2), keepdims=True)
    na = np.sqrt((a * a).sum(axis=(1, 2))) + 1e-6
    corr = (a * c).sum(axis=(1, 2)) / (na * nc)
    k = int(corr.argmax())
    n = 2 * s + 1
    return float(corr[k]), k // n - s, k % n - s


class _RefRolling:
    def __init__(self, win, min_samples, eps=1e-3):
        self._buf = deque(maxlen=win)
        self._min = min_samples
        self._eps = eps

    @staticmethod
    def _median_sorted(a):
        a.sort()
        n = a.size
        return 0.5 * (a[(n - 1) // 2] + a[n // 2])

    def score(self, x):
        if len(self._buf) < self._min:
            self._buf.append(x)
            return 0.0
        a = np.fromiter(self._buf, np.float32)
        med = self._median_sorted(a)
        z = (x - med) / (1.4826 * self._median_sorted(np.abs(a - med)) + self._eps)
        self._buf.append(x)
        return float(z)


class _RefAnalyzer:
    """StreamAnalyzer.update as first written: every statistic recomputed per frame."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._prev_luma = self._prev_color = self._prev_V = self._prev_ovec = None
        self._l1 = deque(maxlen=max(1, cfg.freeze_win))
        self._roll = _RefRolling(cfg.roll_win, cfg.robust_min)
        self._vhist = deque(maxlen=cfg.flicker_win)
        self._han = np.hanning(cfg.flicker_win).astype(np.float32)
        self._cut_cd = self._lock = 0
        self._s2 = self._s1 = 0.0
        self._o1 = False
        self._idx = self._idx_prev = -1

    def _reset(self):
        self._prev_luma = self._prev_color = self._prev_V = self._prev_ovec = None
        self._l1.clear()
        self._vhist.clear()
        self._s2 = self._s1 = 0.0
        self._o1 = False

    @staticmethod
    def _affine(prev, cur):
        pc, cc = prev - prev.mean(), cur - cur.mean()
        vp = float((pc * pc).mean())
        if vp < 1e-6:
            return 1.0, float(cur.mean() - prev.mean()), cur - prev
        a = float((pc * cc).mean()) / vp
        b = float(cur.mean() - a * prev.mean())
        return a, b, cur - (a * prev + b)

    def _frozen(self, luma, V, resid_prev):
        eps = self.cfg.freeze_eps
        newest = len(self._l1) - 1
        for i, (pl, pv) in enumerate(self._l1):
            resid = resid_prev if i == newest else self._affine(pl, luma)[2]
            if float(np.sqrt((resid**2).mean())) + abs(V - pv) < eps:
                return True
        return False

    def _luma_corr(self, prev, cur):
        c = self.cfg
        if min(float(prev.std()), float(cur.std())) < c.ncc_flattol:
            return 1.0
        if c.fast_static:
            corr0 = _ref_ncc0(prev, cur)
            if corr0 >= c.static_corr:
                return corr0
        return _ref_best_shift(prev, cur, c.shift_search)[0]

    def update(self, fs):
        c = self.cfg
        self._idx += 1
        luma, color, V = fs.v_cell_mean, fs.color_mean, fs.exposure
        ovec = fs.struct_grid[:, :, S.SE_OC : S.SE_OS + 1]
        if fs.blank:
            self._reset()
            return TemporalSignals.none(), None, None
        prev_ovec = self._prev_ovec
        if self._prev_luma is None or prev_ovec is None:
            self._reset()
            self._prev_luma, self._prev_color, self._prev_V = luma, color, V
            self._prev_ovec = ovec
            self._l1.append((luma, V))
            self._vhist.append(V)
            self._idx_prev = self._idx
            return TemporalSignals.none(), None, None
        a, b, resid = self._affine(self._prev_luma, luma)
        ori = np.hypot(
            ovec[:, :, 0] - prev_ovec[:, :, 0], ovec[:, :, 1] - prev_ovec[:, :, 1]
        )
        luma_corr = self._luma_corr(self._prev_luma, luma)
        colour = float(np.linalg.norm(self._prev_color - color)) / (
            255.0 * c.color_maxd
        )
        cut_score = max(1.0 - luma_corr, colour)
        robust = self._roll.score(cut_score)
        outlier = (cut_score > c.cut_dissim) and (robust > c.robust_k)
        peak = self._o1 and (self._s1 > self._s2) and (self._s1 > cut_score)
        cut = peak and self._lock == 0
        cut_frame = self._idx_prev if cut else -1
        self._s2, self._s1, self._o1 = self._s1, cut_score, outlier
        freeze = self._frozen(luma, V, resid) and not cut and self._cut_cd == 0
        self._vhist.append(V)
        hist = np.fromiter(self._vhist, np.float32)
        fade = (
            S.fade_score(hist[-c.fade_win :], c.fade_span)
            if len(hist) >= c.fade_win
            else 0.0
        )
        flicker = (
            S.flicker_score(hist, self._han) if len(hist) >= c.flicker_win else 0.0
        )
        self._prev_luma, self._prev_color, self._prev_V = luma, color, V
        self._prev_ovec = ovec
        self._l1.append((luma, V))
        self._idx_prev = self._idx
        self._cut_cd = 1 if cut else max(0, self._cut_cd - 1)
        self._lock = c.min_scene_len if cut else max(0, self._lock - 1)
        return (
            TemporalSignals(
                luma_corr, a, b, cut, cut_score, cut_frame, freeze, fade, flicker
            ),
            resid,
            ori,
        )


# --- inputs: real gate output on synthetic footage ------------------------------------------


def _footage(seed=0, hw=(180, 320)):
    """Duplicates, blank frames, a fade, a cut, a flicker, a pan, noise -- every branch."""
    rng = np.random.default_rng(seed)
    h, w = hw
    base = rng.integers(0, 256, (h, w, 3), np.uint8)
    other = rng.integers(0, 256, (h, w, 3), np.uint8)
    flat = np.full((h, w, 3), 90, np.uint8)
    seq = [base] * 4 + [np.zeros((h, w, 3), np.uint8)] * 2 + [base, base, flat, flat]
    seq += [
        (base.astype(np.float32) * (1 - k / 10)).astype(np.uint8) for k in range(10)
    ]
    seq += [other] * 2
    seq += [
        (other.astype(np.float32) * (0.7 + 0.3 * (k % 2))).astype(np.uint8)
        for k in range(36)
    ]
    seq += [np.roll(other, 3 * k, axis=1) for k in range(12)]
    seq += [rng.integers(0, 256, (h, w, 3), np.uint8) for _ in range(6)]
    return seq


def _bits_equal(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, np.ndarray):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and np.array_equal(a.view(np.uint8), b.view(np.uint8))
        )
    if isinstance(a, float):
        return np.float64(a).view(np.uint64) == np.float64(b).view(np.uint64)
    return a == b


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"freeze_win": 3},
        {"fast_static": False},
        {"shift_search": 0},
        {"shift_search": 5},
    ],
)
def test_update_equals_the_reference_bit_for_bit(overrides):
    cfg = GateConfig(
        feat_threads=1, return_frames=False, skip_duplicates=False, **overrides
    )
    gate = Gate(cfg)
    ref = _RefAnalyzer(cfg)
    try:
        for i, frame in enumerate(_footage()):
            fs, sig = gate.frame(frame)  # the real analyzer ran inside
            resid, ori = fs.residual, fs.ori_change
            rsig, rresid, rori = ref.update(fs)  # (overwrites nothing we still need)
            for k, v in rsig.__dict__.items():
                assert _bits_equal(getattr(sig, k), v), (i, k, getattr(sig, k), v)
            assert _bits_equal(resid, rresid), i
            assert _bits_equal(ori, rori), i
    finally:
        gate.close()


# --- the pieces --------------------------------------------------------------------------


def test_shift_search_equals_best_shift():
    """Every path of ShiftSearch -- the band arrangement with kept arrays, the plain
    version -- gives _best_shift's bits."""
    rng = np.random.default_rng(1)
    search = S.ShiftSearch()
    for g, s in ((64, 3), (64, 1), (32, 2), (64, 3), (16, 0), (128, 3), (16, 4)):
        for _ in range(20):
            prev = (rng.random((g, g)) * 255).astype(np.float32)
            cur = np.roll(
                prev,
                (int(rng.integers(-s, s + 1)), int(rng.integers(-s, s + 1))),
                (0, 1),
            )
            cur = (cur + rng.normal(0, 3, cur.shape)).astype(np.float32)
            want = S._best_shift(prev, cur, s) if s > 0 else (S.ncc0(prev, cur), 0, 0)
            got = search(prev, cur, s)
            assert got[1:] == want[1:] and _bits_equal(got[0], want[0]), (g, s)
            assert _bits_equal(S.best_shift(prev, cur, s)[0], want[0])
    f64 = rng.random((64, 64))  # a float64 map takes the plain path
    assert _bits_equal(
        search(f64, f64 * 1.01, 3)[0], S._best_shift(f64, f64 * 1.01, 3)[0]
    )
    strided = (rng.random((64, 64, 3)) * 255).astype(np.float32)[
        :, :, 1
    ]  # a view like v_cell_mean
    assert _bits_equal(
        search(strided, strided.copy(), 3)[0],
        S._best_shift(strided, strided.copy(), 3)[0],
    )


def test_band_shift_search_equals_best_shift():
    """The band arrangement sums each window in the order the window copy is summed in,
    so _best_shift's bits, on maps of every kind the gate produces and at every size."""
    rng = np.random.default_rng(5)
    search = S.ShiftSearch()
    for g, s in ((64, 3), (64, 1), (64, 5), (32, 2), (24, 1), (128, 3), (16, 4)):
        for kind in range(12):
            grid = (rng.random((g, g, 3, 4)) * 255).astype(np.float32)
            prev = grid[:, :, 2, 0]  # a channel plane, as FrameStats.v_cell_mean
            if kind % 4 == 0:
                cur = (grid * 0.99 + 1).astype(np.float32)[:, :, 2, 0]
            elif kind % 4 == 1:
                dy, dx = (int(rng.integers(-s, s + 1)) for _ in range(2))
                cur = np.roll(prev, (dy, dx), (0, 1))
            elif kind % 4 == 2:
                cur = (prev + rng.normal(0, 3, prev.shape)).astype(np.float32)
            else:
                cur = np.full((g, g), 42.0, np.float32)  # flat: the 1e-6 floors
            want = S._best_shift(prev, cur, s)
            got = search(prev, cur, s)
            assert got[1:] == want[1:] and _bits_equal(got[0], want[0]), (g, s, kind)
            assert _bits_equal(search(np.ascontiguousarray(prev), cur, s)[0], want[0])


def test_luma_statistics_reproduce_the_recomputed_values():
    rng = np.random.default_rng(2)
    for _ in range(50):
        grid = (rng.random((64, 64, 3, 4)) * 255).astype(np.float32)
        prev, cur = grid[:, :, 2, 0], (grid[:, :, 1, 0] * 0.9 + 7).astype(np.float32)
        if rng.random() < 0.2:
            prev = np.full((64, 64), 42.0, np.float32)  # the flat branch of the fit
        p, c = _Luma(prev), _Luma(cur)
        cross = (p.centred * c.centred).sum()
        a, b, resid = StreamAnalyzer._affine_of(p, c, cross)
        ra, rb, rresid = _RefAnalyzer._affine(prev, cur)
        assert _bits_equal(a, ra) and _bits_equal(b, rb) and _bits_equal(resid, rresid)
        assert _bits_equal(StreamAnalyzer._std_of(p), float(prev.std()))
        assert _bits_equal(StreamAnalyzer._std_of(c), float(cur.std()))
        corr0 = float(cross / (np.sqrt(p.sumsq * c.sumsq) + 1e-6))
        assert _bits_equal(corr0, _ref_ncc0(prev, cur))


def test_rolling_robust_ring_equals_the_deque():
    rng = np.random.default_rng(3)
    for win, lo in ((20, 8), (5, 2), (1, 1), (7, 7)):
        ring, ref = _RollingRobust(win, lo), _RefRolling(win, lo)
        for _ in range(200):
            x = float(rng.random() * (10 if rng.random() < 0.1 else 1))
            assert _bits_equal(ring.score(x), ref.score(x)), (win, lo)


def test_v_history_ring_equals_the_deque():
    rng = np.random.default_rng(4)
    for win in (32, 8, 1):
        ring, ref = _VHistory(win), deque(maxlen=win)
        for i in range(3 * win + 5):
            v = float(rng.random() * 255)
            ring.append(v)
            ref.append(v)
            assert len(ring) == len(ref)
            for k in (1, min(len(ref), 4), len(ref)):
                assert np.array_equal(
                    ring.last(k), np.fromiter(ref, np.float32)[-k:]
                ), (win, i, k)
            if i == win + 2:
                ring.clear()
                ref.clear()
