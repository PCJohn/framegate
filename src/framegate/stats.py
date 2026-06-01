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

_PROPOSAL_MAPS = ("V_mean", "V_var", "H_mean", "H_var", "S_mean", "S_var", "saliency", "fine_texture")


@dataclass
class FrameStats:
    """Everything a single frame yields. Raw central moments [mean, var, m3, m4].
    Signals are lazy properties, so callers pay only for what they read."""
    chan: np.ndarray            # (3, 4) per-channel H/S/V moments
    grid: np.ndarray            # (G, G, C, 4) [row, col, channel, moment]
    blank: bool
    shape: tuple                # source (H, W), so spatial outputs map to pixels
    cfg: GateConfig
    thumb: Optional[np.ndarray] = None   # resized input (BGR or gray), if cfg.return_frames
    hsv: Optional[np.ndarray] = None     # its HSV, if cfg.return_frames
    residual: Optional[np.ndarray] = None  # (G,G) photometric change vs the previous frame;
    #   set by StreamAnalyzer, None for a standalone image or the first/post-blank frame

    # --- per-channel grids (views; raw moments) ---
    @property
    def grid_H(self): return self.grid[:, :, S.CH_H, :]
    @property
    def grid_S(self): return self.grid[:, :, S.CH_S, :]
    @property
    def grid_V(self): return self.grid[:, :, S.CH_V, :]

    @property
    def v_cell_mean(self): return self.grid[:, :, S.CH_V, S.M_MEAN]
    @property
    def v_cell_var(self): return self.grid[:, :, S.CH_V, S.M_VAR]

    # --- generic single-frame signals ---
    @property
    def exposure(self): return float(self.chan[S.CH_V, S.M_MEAN])
    @property
    def contrast(self): return float(np.sqrt(max(self.chan[S.CH_V, S.M_VAR], 0.0)))
    @property
    def colorfulness(self): return float(self.chan[S.CH_S, S.M_MEAN])   # ~0 -> grayscale/graphic
    @property
    def detail(self): return float(self.v_cell_var.mean())              # SI-like spatial complexity
    @property
    def flat_fraction(self): return float((self.v_cell_var < self.cfg.solid_thresh).mean())
    @property
    def noise_floor(self):
        """Std of the flattest cell ~= sensor / compression noise floor."""
        return float(np.sqrt(max(self.v_cell_var.min(), 0.0)))
    @property
    def clipping(self):
        """Exposure asymmetry from V skew. >0: piled near black (crushed shadows);
        <0: piled near white (blown highlights); ~0: balanced."""
        sd = self.chan[S.CH_V, S.M_VAR] ** 0.5
        return float(self.chan[S.CH_V, S.M_M3] / sd ** 3) if sd > 1e-6 else 0.0

    # --- derived maps (cached: pure per-frame, so a duplicate frame reuses them) ---
    @cached_property
    def saliency(self):
        return S.saliency_map(self.grid_V, self.grid_S)

    @cached_property
    def fine_texture(self):
        """(G,G) fine high-frequency achromatic texture -- a generic text/print/UI
        cue (not OCR). See signals.fine_texture."""
        c = self.cfg
        return S.fine_texture(self.grid_V, self.grid_S, c.ftex_achromatic_w, c.ftex_coarse_k, c.ftex_line_k)

    @property
    def rois(self):
        """Region proposals as a single labelled list: [(box, [labels]), ...]. Each box
        (x0,y0,x1,y1) is in source-frame pixels, ready to crop -- frame[y0:y1, x0:x1].
        Labels name the maps that proposed the region (H/S/V mean+variance, saliency,
        fine_texture; '_inv' = a map's dark side; on video also 'motion'), so len(labels)
        reflects how many maps agree. Built in one pass: each map is border-trimmed in both
        polarities, full-frame (non-localizing) boxes are dropped, and overlapping boxes are
        merged (cfg.roi_merge_iou). The list is unordered; empty if blank or nothing localizes."""
        if self.blank:
            return []
        g = self.grid
        stack = np.stack([g[:, :, S.CH_V, S.M_MEAN], g[:, :, S.CH_V, S.M_VAR],
                          g[:, :, S.CH_H, S.M_MEAN], g[:, :, S.CH_H, S.M_VAR],
                          g[:, :, S.CH_S, S.M_MEAN], g[:, :, S.CH_S, S.M_VAR],
                          self.saliency, self.fine_texture])
        hi, lo = S.roi_boxes(stack, self.shape, self.cfg.roi_k)
        limit = 0.95 * self.shape[0] * self.shape[1]
        boxes, labels = [], []
        for name, bh, bl in zip(_PROPOSAL_MAPS, hi, lo):
            for tag, b in ((name, bh), (name + "_inv", bl)):
                if b and (b[2] - b[0]) * (b[3] - b[1]) < limit:
                    boxes.append(b); labels.append(tag)
        if self.motion is not None:                       # video: add a moving-region box (hi side only)
            mb = S.roi_boxes(self.motion[None], self.shape, self.cfg.roi_k)[0][0]
            if mb and (mb[2] - mb[0]) * (mb[3] - mb[1]) < limit:
                boxes.append(mb); labels.append("motion")
        return S.merge_boxes(boxes, labels, self.cfg.roi_merge_iou)

    @property
    def motion(self):
        """(G,G) motion magnitude = |residual| (change after removing global gain/bias),
        or None for a standalone image / first / post-blank frame. The single motion
        quantity, used by both the visualizer and the 'motion' ROI proposal."""
        return None if self.residual is None else np.abs(self.residual)

    @cached_property
    def color_mean(self) -> np.ndarray:
        """Global saturation + saturation-weighted hue vector [S, S*cos2H, S*sin2H],
        averaged over cells. Unsaturated cells (hue = noise) contribute ~nothing."""
        sat = self.grid_S[:, :, S.M_MEAN]
        ang = self.grid_H[:, :, S.M_MEAN] * (np.pi / 90.0)   # OpenCV hue 0..180 -> 0..2pi
        return np.array([sat.mean(), (sat * np.cos(ang)).mean(), (sat * np.sin(ang)).mean()], np.float32)


class FrameGate:
    """Per-frame extractor. Owns reusable buffers and one StatsComputer; no
    temporal state, so it works identically on a still image or a video frame.
    Accepts BGR (H,W,3) or grayscale (H,W)/(H,W,1) uint8 input. Not thread-safe
    (the scratch buffers are reused per call); use one FrameGate per stream."""

    def __init__(self, cfg: GateConfig = None):
        self.cfg = cfg or GateConfig()
        t = self.cfg.thumb
        self._fast = cv2.FastFeatureDetector_create(threshold=self.cfg.fast_thresh,
                                                     nonmaxSuppression=False)
        self._stats = ts.StatsComputer(
            shape=(t, t, 3),
            axes=[(0, 1)],
            stride=(self.cfg.stride, self.cfg.stride, 1),
            grid=(self.cfg.grid_exp, self.cfg.grid_exp, 2),
        )
        self._bgr = np.empty((t, t, 3), np.uint8)    # scratch reused when not returning frames
        self._gray = np.empty((t, t), np.uint8)
        self._hsv = np.empty((t, t, 3), np.uint8)

    def _to_hsv(self, frame: np.ndarray, keep: bool):
        """Resize to the thumbnail and produce HSV. Grayscale becomes H=S=0, V=luma,
        so colour signals correctly read as zero. With `keep`, outputs are fresh
        arrays the caller can hold; otherwise reused scratch buffers."""
        t = self.cfg.thumb
        hsv = np.empty((t, t, 3), np.uint8) if keep else self._hsv
        if frame.ndim == 2 or frame.shape[2] == 1:
            thumb = np.empty((t, t), np.uint8) if keep else self._gray
            cv2.resize(frame.reshape(frame.shape[0], frame.shape[1]), (t, t),
                       dst=thumb, interpolation=cv2.INTER_NEAREST)
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
        grid = r["grid"].astype(np.float32)

        # Lossless: a stats-flat frame has no FAST corners, so skip the detector.
        blank = (float(grid[:, :, S.CH_V, S.M_VAR].max()) < self.cfg.solid_thresh
                 or len(self._fast.detect(hsv[:, :, S.CH_V], None)) == 0)

        return FrameStats(chan=chan, grid=grid, blank=blank, shape=(h, w), cfg=self.cfg,
                          thumb=thumb, hsv=hsv if keep else None)
