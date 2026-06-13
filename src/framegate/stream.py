"""Temporal layer: StreamAnalyzer consumes FrameStats in order and emits
TemporalSignals (cut, freeze, fade, flicker; the motion map lives on FrameStats).
Reset across blank frames,
since you can't diff a frame against content from before a hard break.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

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
    dispatch dominates at this window size); the value is identical."""

    def __init__(self, win, min_samples, eps=1e-3):
        self._buf = deque(maxlen=win)
        self._min = min_samples
        self._eps = eps

    @staticmethod
    def _median_sorted(a):
        a.sort()
        n = a.size
        return 0.5 * (a[(n - 1) // 2] + a[n // 2])

    def score(self, x: float) -> float:
        if len(self._buf) < self._min:
            s = 0.0
        else:
            a = np.fromiter(self._buf, np.float32)
            med = self._median_sorted(a)
            s = (x - med) / (1.4826 * self._median_sorted(np.abs(a - med)) + self._eps)
        self._buf.append(x)
        return float(s)


class StreamAnalyzer:
    """Feed it FrameStats in order; returns TemporalSignals.

    Cut = max of two complementary, motion-robust dissimilarities (so either can
    trigger): 1 - motion-compensated luma correlation (structural/spatial cuts,
    pan-robust, blind to flat colour) and a normalized global saturation+hue-vector
    shift (equiluminant/flat colour cuts). A cut fires only when that score is a
    robust (median+MAD) outlier AND an isolated peak (rejecting pans/dissolves),
    debounced by a minimum shot length, confirmed with 1 frame of latency."""

    def __init__(self, cfg: GateConfig = None):
        self.cfg = cfg or GateConfig()
        self._prev_luma: Optional[np.ndarray] = None  # prev cell-mean luma map (G, G)
        self._prev_color: Optional[np.ndarray] = None  # prev global colour vector (3,)
        self._prev_V: Optional[float] = None
        self._roll = _RollingRobust(self.cfg.roll_win, self.cfg.robust_min)
        self._vhist = deque(maxlen=self.cfg.flicker_win)
        self._han = np.hanning(self.cfg.flicker_win).astype(
            np.float32
        )  # precomputed for flicker
        self._cut_cd = 0  # suppress freeze 1 frame post-cut
        self._lock = 0  # min-shot-length cut debounce
        self._s2 = self._s1 = 0.0
        self._o1 = False
        self._idx = -1
        self._idx_prev = -1

    def _reset(self):
        self._prev_luma = self._prev_color = self._prev_V = None
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

    def _luma_corr(self, prev, cur):
        """Motion-compensated luma correlation. Skips the shift search on near-static
        frames (zero-shift corr already >= static_corr), where the search cannot
        change the cut decision -- effectively lossless."""
        c = self.cfg
        if min(float(prev.std()), float(cur.std())) < c.ncc_flattol:
            return 1.0
        if c.fast_static:
            a, b = prev - prev.mean(), cur - cur.mean()
            corr0 = float(
                (a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6)
            )
            if corr0 >= c.static_corr:
                return corr0
        return S.best_shift(prev, cur, c.shift_search)[0]

    def _cut_score(self, prev_luma, prev_color, cur_luma, cur_color):
        """(cut_score, luma_corr) = max of luma-structure and global colour-shift."""
        c = self.cfg
        luma_corr = self._luma_corr(prev_luma, cur_luma)
        color = float(np.linalg.norm(prev_color - cur_color)) / (255.0 * c.color_maxd)
        return max(c.cut_w_luma * (1.0 - luma_corr), c.cut_w_color * color), luma_corr

    def update(self, fs) -> TemporalSignals:
        c = self.cfg
        self._idx += 1
        luma, color, V = fs.v_cell_mean, fs.color_mean, fs.exposure

        if fs.blank:
            self._reset()
            return TemporalSignals.none()
        if self._prev_luma is None:
            self._reset()
            self._prev_luma, self._prev_color, self._prev_V = luma, color, V
            self._vhist.append(V)
            self._idx_prev = self._idx
            return TemporalSignals.none()

        a, b, resid = self._affine(self._prev_luma.ravel(), luma.ravel())
        resid_rms = float(np.sqrt((resid**2).mean()))
        dV = V - self._prev_V
        fs.residual = resid.reshape(luma.shape)  # annotate the frame with its motion

        cut_score, luma_corr = self._cut_score(
            self._prev_luma, self._prev_color, luma, color
        )
        robust = self._roll.score(cut_score)  # must update EVERY frame
        outlier = (cut_score > c.cut_dissim) and (robust > c.robust_k)
        peak = self._o1 and (self._s1 > self._s2) and (self._s1 > cut_score)
        cut = peak and self._lock == 0
        cut_frame = self._idx_prev if cut else -1
        self._s2, self._s1, self._o1 = self._s1, cut_score, outlier

        freeze = (resid_rms + abs(dV) < c.freeze_eps) and not cut and self._cut_cd == 0

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
        self._idx_prev = self._idx
        self._cut_cd = 1 if cut else max(0, self._cut_cd - 1)
        self._lock = c.min_scene_len if cut else max(0, self._lock - 1)

        return TemporalSignals(
            luma_corr, a, b, cut, cut_score, cut_frame, freeze, fade, flicker
        )
