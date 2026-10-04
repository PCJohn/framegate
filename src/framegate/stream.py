"""Temporal layer: StreamAnalyzer consumes FrameStats in order and emits
TemporalSignals (cut, freeze, fade, flicker; the motion map lives on FrameStats).
Reset across blank frames,
since you can't diff a frame against content from before a hard break.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import signals as S
from .config import GateConfig


@dataclass
class TemporalSignals:
    """Cross-frame scalar signals. The per-frame *maps* (saliency, motion, ...) live on
    FrameStats; this layer carries only scalars derived from the transition between
    frames, so reading them is always cheap."""

    struct_corr: float  # motion-compensated luma correlation
    gain: float  # affine a (global photometric gain, luma)
    bias: float  # affine b
    cut: bool
    cut_score: float
    cut_frame: int  # true frame index of the cut (-1 if none)
    freeze: bool
    fade: float  # signed fade strength (-1 out .. +1 in)
    flicker: float  # periodic-power fraction (0..1)

    @classmethod
    def none(cls) -> "TemporalSignals":
        return cls(1.0, 1.0, 0.0, False, 0.0, -1, False, 0.0, 0.0)


class _RollingRobust:
    """Median + MAD over a trailing window, excluding the current sample, so an
    event can't inflate its own baseline. Uses an explicit sort (np.median's
    dispatch dominates at this window size); the value is identical.

    The window is a ring in one float32 array: the median and MAD sort a copy, so the
    ring's order does not matter, and no array is built from a deque per frame."""

    def __init__(self, win, min_samples, eps=1e-3):
        self._ring = np.empty(win, np.float32)
        self._win = win
        self._n = 0  # samples recorded so far (the ring holds min(_n, win) of them)
        self._min = min_samples
        self._eps = eps

    @staticmethod
    def _median_sorted(a):
        a.sort()
        n = a.size
        return 0.5 * (a[(n - 1) // 2] + a[n // 2])

    def _record(self, x: float) -> None:
        self._ring[self._n % self._win] = x
        self._n += 1

    def score(self, x: float) -> float:
        """Robust z-score of x against the trailing window, then record x. 0.0 until the
        window has min_samples (baseline not yet trusted). x is scored against the window
        *before* being added, so an event can't inflate its own baseline."""
        n = self._n
        if n < self._min:
            self._record(x)
            return 0.0
        a = self._ring[: min(n, self._win)].copy()
        med = self._median_sorted(a)
        z = (x - med) / (1.4826 * self._median_sorted(np.abs(a - med)) + self._eps)
        self._record(x)
        return float(z)


class _Luma:
    """One frame's cell-mean luma map with the statistics every pairwise step reads: the
    mean, the centred map and its sum of squares. Computed once when the frame is current
    and carried over when it becomes the previous frame, so nothing is derived twice --
    the same numpy operations, so the same bits as recomputing them."""

    __slots__ = ("centred", "map", "mean", "sumsq")

    def __init__(self, luma: np.ndarray):
        self.map = luma
        self.mean = luma.mean()
        self.centred = luma - self.mean
        self.sumsq = (self.centred * self.centred).sum()


class _VHistory:
    """The last `win` brightness values in order, as one contiguous slice of a doubled
    ring (each value is written twice, `win` apart), so no array is built per frame."""

    def __init__(self, win: int):
        self._win = win
        self._buf = np.empty(2 * win, np.float32)
        self._n = 0

    def clear(self) -> None:
        self._n = 0

    def append(self, v: float) -> None:
        i = self._n % self._win
        self._buf[i] = v
        self._buf[i + self._win] = v
        self._n += 1

    def __len__(self) -> int:
        return min(self._n, self._win)

    def last(self, k: int) -> np.ndarray:
        """The newest k values, oldest first (k <= len(self))."""
        end = self._n % self._win + self._win
        return self._buf[end - k : end]


class StreamAnalyzer:
    """Feed it FrameStats in order; returns TemporalSignals.

    Cut = max of two complementary, motion-robust dissimilarities (so either can
    trigger): 1 - motion-compensated luma correlation (structural/spatial cuts,
    pan-robust, blind to flat colour) and a normalized global saturation+hue-vector
    shift (equiluminant/flat colour cuts). A cut fires only when that score is a
    robust (median+MAD) outlier AND an isolated peak (rejecting pans/dissolves),
    debounced by a minimum shot length, confirmed with 1 frame of latency."""

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self._prev_luma: np.ndarray | None = None  # prev cell-mean luma map (G, G)
        self._prev: _Luma | None = None  # ... with its statistics (see _Luma)
        self._prev_color: np.ndarray | None = None  # prev global colour vector (3,)
        self._prev_ovec: np.ndarray | None = (
            None  # prev per-cell orientation vec (G,G,2)
        )
        self._prev_V: float | None = None
        self._l1: deque = deque(maxlen=max(1, self.cfg.freeze_win))  # (luma, V) ring
        self._roll = _RollingRobust(self.cfg.roll_win, self.cfg.robust_min)
        self._vhist = _VHistory(self.cfg.flicker_win)
        self._han = np.hanning(self.cfg.flicker_win).astype(
            np.float32
        )  # precomputed for flicker
        self._shift = S.ShiftSearch()  # best_shift with its working arrays kept
        self._cut_cd = 0  # suppress freeze 1 frame post-cut
        self._lock = 0  # min-shot-length cut debounce
        self._s2 = self._s1 = 0.0
        self._o1 = False
        self._idx = -1
        self._idx_prev = -1

    def _reset(self):
        self._prev_luma = self._prev_color = self._prev_V = None
        self._prev = self._prev_ovec = None
        self._l1.clear()
        self._vhist.clear()
        self._s2 = self._s1 = 0.0
        self._o1 = False

    @staticmethod
    def _affine(prev, cur):
        """Fit cur ~= a*prev + b over luma cells (zero shift). Returns
        (a, b, residual); the residual is the photometric-normalized change map."""
        pc, cc = prev - prev.mean(), cur - cur.mean()
        vp = float((pc * pc).mean())
        if vp < 1e-6:
            return 1.0, float(cur.mean() - prev.mean()), cur - prev
        a = float((pc * cc).mean()) / vp
        b = float(cur.mean() - a * prev.mean())
        return a, b, cur - (a * prev + b)

    def _frozen(self, luma, V, resid_prev) -> bool:
        """L1: does this frame affine-match any frame in the ring? The affine fit
        absorbs a global brightness/contrast change; |dV| catches an exposure shift the
        fit's own recentring would hide. The newest ring entry is t-1, whose fit the
        caller already computed for the motion annotation -- pass it as `resid_prev` so
        the common freeze_win=1 case does no extra fit at all."""
        eps = self.cfg.freeze_eps
        newest = len(self._l1) - 1
        for i, (pl, pv) in enumerate(self._l1):
            resid = resid_prev if i == newest else self._affine(pl, luma)[2]
            if float(np.sqrt((resid**2).mean())) + abs(V - pv) < eps:
                return True
        return False

    # The pairwise steps below are _affine(), ncc0() and the two std() calls written on the
    # carried statistics (_Luma): each formula is the original's, operation for operation,
    # with the reductions it would recompute replaced by the ones already taken on the
    # same arrays. `n` is the cell count; a mean is numpy's float32 sum over float32 n.

    @staticmethod
    def _affine_of(prev: _Luma, cur: _Luma, cross):
        """_affine(prev.map, cur.map) from the statistics; `cross` is (pc * cc).sum()."""
        n = prev.map.size
        vp = float(np.true_divide(prev.sumsq, n, dtype=np.float32))  # (pc * pc).mean()
        if vp < 1e-6:
            return 1.0, float(cur.mean - prev.mean), cur.map - prev.map
        a = (
            float(np.true_divide(cross, n, dtype=np.float32)) / vp
        )  # (pc * cc).mean() / vp
        b = float(cur.mean - a * prev.mean)
        return a, b, cur.map - (a * prev.map + b)

    @staticmethod
    def _std_of(x: _Luma) -> float:
        """float(x.map.std()): the root of the mean of the centred squares, in float32."""
        return float(np.sqrt(np.true_divide(x.sumsq, x.map.size, dtype=np.float32)))

    def _luma_corr(self, prev: _Luma, cur: _Luma, cross):
        """Motion-compensated luma correlation. Skips the shift search on near-static
        frames (zero-shift corr already >= static_corr), where the search cannot
        change the cut decision -- effectively lossless."""
        c = self.cfg
        if min(self._std_of(prev), self._std_of(cur)) < c.ncc_flattol:
            return 1.0
        if c.fast_static:  # S.ncc0(prev.map, cur.map), from the statistics
            corr0 = float(cross / (np.sqrt(prev.sumsq * cur.sumsq) + 1e-6))
            if corr0 >= c.static_corr:
                return corr0
        return self._shift(prev.map, cur.map, c.shift_search)[0]

    def _cut_score(
        self, prev: _Luma, prev_color, cur: _Luma, cur_color, cross
    ) -> tuple:
        """(cut_score, luma_corr) = max of the luma-structure and global colour-shift
        dissimilarities, each already normalized to ~[0, 1]."""
        c = self.cfg
        luma_corr = self._luma_corr(prev, cur, cross)
        color = float(np.linalg.norm(prev_color - cur_color)) / (255.0 * c.color_maxd)
        return max(1.0 - luma_corr, color), luma_corr

    def update(self, fs) -> TemporalSignals:
        c = self.cfg
        self._idx += 1
        luma, color, V = fs.v_cell_mean, fs.color_mean, fs.exposure
        ovec = fs.struct_grid[:, :, S.SE_OC : S.SE_OS + 1]

        if fs.blank:
            self._reset()
            return TemporalSignals.none()
        prev_ovec = self._prev_ovec  # set and cleared together with _prev_luma
        prev = self._prev
        cur = _Luma(luma)  # its statistics, taken once; carried as `prev` next frame
        if prev is None or prev_ovec is None:
            self._reset()
            self._prev_luma, self._prev_color, self._prev_V = luma, color, V
            self._prev, self._prev_ovec = cur, ovec
            self._l1.append((luma, V))
            self._vhist.append(V)
            self._idx_prev = self._idx
            return TemporalSignals.none()

        cross = (
            prev.centred * cur.centred
        ).sum()  # shared by the fit and the correlation
        a, b, resid = self._affine_of(prev, cur, cross)  # 2-D, no ravel copies
        fs.residual = resid  # already (G,G); annotate the frame with its motion vs t-1
        fs.ori_change = np.hypot(
            ovec[:, :, 0] - prev_ovec[:, :, 0],
            ovec[:, :, 1] - prev_ovec[:, :, 1],
        )

        cut_score, luma_corr = self._cut_score(
            prev, self._prev_color, cur, color, cross
        )
        robust = self._roll.score(cut_score)  # must update EVERY frame
        outlier = (cut_score > c.cut_dissim) and (robust > c.robust_k)
        peak = self._o1 and (self._s1 > self._s2) and (self._s1 > cut_score)
        cut = peak and self._lock == 0
        cut_frame = self._idx_prev if cut else -1
        self._s2, self._s1, self._o1 = self._s1, cut_score, outlier

        # L1: frozen if this frame affine-matches any recent kept frame. The newest ring
        # entry is t-1, whose residual we already have; only a wider window costs extra.
        freeze = self._frozen(luma, V, resid) and not cut and self._cut_cd == 0

        self._vhist.append(V)
        n_hist = len(self._vhist)
        fade = (
            S.fade_score(self._vhist.last(c.fade_win), c.fade_span)
            if n_hist >= c.fade_win
            else 0.0
        )
        flicker = (
            S.flicker_score(self._vhist.last(c.flicker_win), self._han)
            if n_hist >= c.flicker_win
            else 0.0
        )

        self._prev_luma, self._prev_color, self._prev_V = luma, color, V
        self._prev, self._prev_ovec = cur, ovec
        self._l1.append((luma, V))
        self._idx_prev = self._idx
        self._cut_cd = 1 if cut else max(0, self._cut_cd - 1)
        self._lock = c.min_scene_len if cut else max(0, self._lock - 1)

        return TemporalSignals(
            luma_corr, a, b, cut, cut_score, cut_frame, freeze, fade, flicker
        )
