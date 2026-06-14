"""Per-frame extraction: FrameStats (the stateless descriptor) and FrameGate
(the extractor that produces it). Works on a single image or a video frame; for
video the temporal layer (StreamAnalyzer) consumes a stream of FrameStats.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import cv2
import numpy as np

import tensorstats as ts

from . import signals as S
from .config import GateConfig


@dataclass
class FrameStats:
    """Everything a single frame yields. Raw central moments [mean, var, m3, m4].
    Signals are lazy properties, so callers pay only for what they read."""

    chan: np.ndarray  # (3, 4) per-channel H/S/V moments
    grid: (
        np.ndarray
    )  # (G, G, C, 4) [row, col, channel, moment] -- finest level (== grids[0])
    blank: bool
    shape: tuple  # source (H, W), so spatial outputs map to pixels
    cfg: GateConfig
    grids: (
        tuple
    ) = ()  # pyramid levels finest->coarsest, each (G_k, G_k, C, 4); grids[0] is grid
    thumb: Optional[np.ndarray] = (
        None  # resized input (BGR or gray), if cfg.return_frames
    )
    hsv: Optional[np.ndarray] = None  # its HSV, if cfg.return_frames
    residual: Optional[np.ndarray] = (
        None  # (G,G) photometric change vs the previous frame;
    )
    #   set by StreamAnalyzer, None for a standalone image or the first/post-blank frame

    # --- per-channel grids (views; raw moments) ---
    @property
    def grid_H(self):
        return self.grid[:, :, S.CH_H, :]

    @property
    def grid_S(self):
        return self.grid[:, :, S.CH_S, :]

    @property
    def grid_V(self):
        return self.grid[:, :, S.CH_V, :]

    @property
    def v_cell_mean(self):
        return self.grid[:, :, S.CH_V, S.M_MEAN]

    @property
    def v_cell_var(self):
        return self.grid[:, :, S.CH_V, S.M_VAR]

    # --- generic single-frame signals ---
    @property
    def exposure(self):
        return float(self.chan[S.CH_V, S.M_MEAN])

    @property
    def contrast(self):
        return float(np.sqrt(max(self.chan[S.CH_V, S.M_VAR], 0.0)))

    @property
    def colorfulness(self):
        return float(self.chan[S.CH_S, S.M_MEAN])  # ~0 -> grayscale/graphic

    @property
    def detail(self):
        return float(self.v_cell_var.mean())  # SI-like spatial complexity

    @property
    def flat_fraction(self):
        return float((self.v_cell_var < self.cfg.solid_thresh).mean())

    @property
    def noise_floor(self):
        """Std of the flattest cell ~= sensor / compression noise floor."""
        return float(np.sqrt(max(self.v_cell_var.min(), 0.0)))

    @property
    def clipping(self):
        """Exposure asymmetry from V skew. >0: piled near black (crushed shadows);
        <0: piled near white (blown highlights); ~0: balanced."""
        sd = self.chan[S.CH_V, S.M_VAR] ** 0.5
        return float(self.chan[S.CH_V, S.M_M3] / sd**3) if sd > 1e-6 else 0.0

    # --- derived maps (cached: pure per-frame, so a duplicate frame reuses them) ---
    @cached_property
    def saliency(self):
        return S.saliency_map(self.grid_V, self.grid_S, self.cfg.sal_surround)

    @cached_property
    def fine_texture(self):
        """(G,G) fine high-frequency achromatic texture -- a generic text/print/UI
        cue (not OCR). See signals.fine_texture."""
        c = self.cfg
        return S.fine_texture(
            self.grid_V,
            self.grid_S,
            c.ftex_achromatic_w,
            c.ftex_coarse_k,
            c.ftex_line_k,
        )

    @property
    def motion(self):
        """(G,G) motion magnitude vs the previous frame -- |residual| after removing global
        gain/bias -- or None for a standalone image / first / post-blank frame. A noise floor
        is subtracted: the larger of a relative local floor (cfg.motion_floor_k * local mean
        |residual| over a cfg.motion_surround neighborhood) and an absolute floor
        (cfg.motion_abs_floor grey levels). The local term adapts to regionally-varying noise;
        the absolute term ensures sub-grey-level change reads as no motion on any frame. Set
        both motion_floor_k and motion_abs_floor to 0 for the raw magnitude; the signed change
        is always in `residual`."""
        if self.residual is None:
            return None
        m = np.abs(self.residual)
        floor = self.cfg.motion_abs_floor
        if self.cfg.motion_floor_k > 0.0:
            floor = np.maximum(
                self.cfg.motion_floor_k
                * S.box(m, self.cfg.motion_surround, self.cfg.motion_surround),
                floor,
            )
        return np.maximum(m - floor, 0.0)

    @cached_property
    def color_mean(self) -> np.ndarray:
        """Global saturation + saturation-weighted hue vector [S, S*cos2H, S*sin2H],
        averaged over cells. Unsaturated cells (hue = noise) contribute ~nothing."""
        sat = self.grid_S[:, :, S.M_MEAN]
        ang = self.grid_H[:, :, S.M_MEAN] * (
            np.pi / 90.0
        )  # OpenCV hue 0..180 -> 0..2pi
        return np.array(
            [sat.mean(), (sat * np.cos(ang)).mean(), (sat * np.sin(ang)).mean()],
            np.float32,
        )


class FrameGate:
    """Per-frame extractor. Owns reusable buffers and one StatsComputer; no
    temporal state, so it works identically on a still image or a video frame.
    Accepts BGR (H,W,3) or grayscale (H,W)/(H,W,1) uint8 input. Not thread-safe
    (the scratch buffers are reused per call); use one FrameGate per stream."""

    def __init__(self, cfg: GateConfig = None):
        self.cfg = cfg or GateConfig()
        t = self.cfg.thumb
        self._fast = cv2.FastFeatureDetector_create(
            threshold=self.cfg.fast_thresh, nonmaxSuppression=False
        )
        self._stats = ts.StatsComputer(
            shape=(t, t, 3),
            axes=[(0, 1)],
            stride=(self.cfg.stride, self.cfg.stride, 1),
            grid=[(e, e, 2) for e in self.cfg.pyramid_exps],
        )
        self._bgr = np.empty(
            (t, t, 3), np.uint8
        )  # scratch reused when not returning frames
        self._gray = np.empty((t, t), np.uint8)
        self._hsv = np.empty((t, t, 3), np.uint8)
        # FAST runs on a subsample of V; precompute the step + a reusable buffer.
        self._fast_step = max(1, t // self.cfg.fast_thumb)
        if self._fast_step > 1:
            n = len(range(0, t, self._fast_step))
            self._vfast = np.empty((n, n), np.uint8)

    def _to_hsv(self, frame: np.ndarray, keep: bool):
        """Resize to the thumbnail and produce HSV. Grayscale becomes H=S=0, V=luma,
        so colour signals correctly read as zero. With `keep`, outputs are fresh
        arrays the caller can hold; otherwise reused scratch buffers."""
        t = self.cfg.thumb
        hsv = np.empty((t, t, 3), np.uint8) if keep else self._hsv
        if frame.ndim == 2 or frame.shape[2] == 1:
            thumb = np.empty((t, t), np.uint8) if keep else self._gray
            cv2.resize(
                frame.reshape(frame.shape[0], frame.shape[1]),
                (t, t),
                dst=thumb,
                interpolation=cv2.INTER_NEAREST,
            )
            hsv[:, :, :2] = 0
            hsv[:, :, 2] = thumb
        else:
            thumb = np.empty((t, t, 3), np.uint8) if keep else self._bgr
            cv2.resize(frame, (t, t), dst=thumb, interpolation=cv2.INTER_NEAREST)
            cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV, dst=hsv)
        return hsv, (thumb if keep else None)

    def process(self, frame: np.ndarray) -> FrameStats:
        h, w = frame.shape[:2]
        keep = self.cfg.return_frames
        hsv, thumb = self._to_hsv(frame, keep)
        r = self._stats.compute(hsv)
        chan = r["0,1"].astype(np.float32)
        grids = tuple(
            r[f"grid_{i}"].astype(np.float32) for i in range(self.cfg.n_levels)
        )
        grid = grids[
            0
        ]  # finest = output-map resolution; coarser levels feed multi-scale signals

        # Lossless: a stats-flat frame has no FAST corners, so skip the detector.
        # FAST is only the blank check, so run it on a cheaper subsample of V.
        if self._fast_step == 1:
            vch = hsv[:, :, S.CH_V]
        else:
            self._vfast[:] = hsv[:: self._fast_step, :: self._fast_step, S.CH_V]
            vch = self._vfast
        blank = (
            float(grid[:, :, S.CH_V, S.M_VAR].max()) < self.cfg.solid_thresh
            or len(self._fast.detect(vch, None)) == 0
        )

        return FrameStats(
            chan=chan,
            grid=grid,
            grids=grids,
            blank=blank,
            shape=(h, w),
            cfg=self.cfg,
            thumb=thumb,
            hsv=hsv if keep else None,
        )
