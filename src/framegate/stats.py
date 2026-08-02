"""Per-frame extraction: FrameStats (the stateless descriptor) and FrameGate
(the extractor that produces it). Works on a single image or a video frame; for
video the temporal layer (StreamAnalyzer) consumes a stream of FrameStats.
"""

from dataclasses import dataclass
from functools import cached_property

import cv2
import imfeat  # type: ignore[import-untyped]  # imfeat needs a py.typed marker
import numpy as np

from . import signals as S
from .config import GateConfig

_F = len(imfeat.FEATURE_NAMES)  # 38 features per channel in a pyramid map
_C = 3  # HSV


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
    thumb: np.ndarray | None = None  # resized input (BGR or gray), if cfg.return_frames
    hsv: np.ndarray | None = None  # its HSV, if cfg.return_frames
    residual: np.ndarray | None = (
        None  # (G,G) photometric change vs the previous frame;
    )
    #   set by StreamAnalyzer, None for a standalone image or the first/post-blank frame
    ori_change: np.ndarray | None = (
        None  # (G,G) ||d(edge orientation vector)|| vs prev; illumination-invariant
    )
    struct: dict | None = (
        None  # imfeat structure maps for V: "grid_0" (cells,cells,5) + "global" (5,)
    )
    phash: int = 0  # whole-frame luma pHash (imfeat), one uint64; shot-memory key

    # --- per-channel grids (views; raw moments) ---
    @property
    def grid_H(self) -> np.ndarray:
        return self.grid[:, :, S.CH_H, :]

    @property
    def grid_S(self) -> np.ndarray:
        return self.grid[:, :, S.CH_S, :]

    @property
    def grid_V(self) -> np.ndarray:
        return self.grid[:, :, S.CH_V, :]

    @property
    def v_cell_mean(self) -> np.ndarray:
        return self.grid[:, :, S.CH_V, S.M_MEAN]

    @property
    def v_cell_var(self) -> np.ndarray:
        return self.grid[:, :, S.CH_V, S.M_VAR]

    # --- generic single-frame signals ---
    @property
    def exposure(self) -> float:
        return float(self.chan[S.CH_V, S.M_MEAN])

    @property
    def contrast(self) -> float:
        return float(np.sqrt(max(self.chan[S.CH_V, S.M_VAR], 0.0)))

    @property
    def colorfulness(self) -> float:
        return float(self.chan[S.CH_S, S.M_MEAN])  # ~0 -> grayscale/graphic

    @property
    def detail(self) -> float:
        return float(self.v_cell_var.mean())  # SI-like spatial complexity

    @property
    def flat_fraction(self) -> float:
        return float((self.v_cell_var < self.cfg.solid_thresh).mean())

    @property
    def noise_floor(self) -> float:
        """Std of the flattest cell ~= sensor / compression noise floor."""
        return float(np.sqrt(max(self.v_cell_var.min(), 0.0)))

    @property
    def clipping(self) -> float:
        """Exposure asymmetry from V skew. >0: piled near black (crushed shadows);
        <0: piled near white (blown highlights); ~0: balanced."""
        sd = self.chan[S.CH_V, S.M_VAR] ** 0.5
        return float(self.chan[S.CH_V, S.M_M3] / sd**3) if sd > 1e-6 else 0.0

    # --- derived maps (cached: pure per-frame, so a duplicate frame reuses them) ---
    @cached_property
    def saliency(self) -> np.ndarray:
        return S.saliency_map(
            self.grid_V, self.grid_S, self.struct_grid, self.cfg.sal_surround
        )

    @cached_property
    def text(self) -> np.ndarray:
        """(G,G) text likelihood from low-level texture: a fine, achromatic,
        coherent, bimodal cue. Tuned for dense/printed text (body text, captions, UI);
        a cue, not OCR. See signals.text."""
        c = self.cfg
        return S.text(
            self.grid_V,
            self.grid_S,
            self.coherence,
            c.text_achromatic_w,
            c.text_coarse_k,
            c.text_line_k,
            c.text_skew_w,
            c.text_skew_ref,
            c.text_coherence_w,
        )

    @property
    def motion(self) -> np.ndarray | None:
        """(G,G) motion magnitude vs the previous frame: |residual| (after removing
        global gain/bias) minus a noise floor, then structurally validated -- or None
        for a still image / first / post-blank frame. The floor is the larger of a
        relative local term (motion_floor_k * local-mean |residual| over motion_surround
        cells) and an absolute term (motion_abs_floor grey levels). Structural validation
        (motion_struct_w) down-weights cells whose luma changed but whose *edge
        orientation* did not: real motion moves edges, whereas a regional lighting/shadow
        shift scales gradients without reorienting them (orientation is illumination-
        invariant), so it is suppressed. Set the floors and motion_struct_w to 0 for the
        raw magnitude; the signed change is always in `residual`."""
        if self.residual is None:
            return None
        m = np.abs(self.residual)
        floor = self.cfg.motion_abs_floor
        if self.cfg.motion_floor_k > 0.0:
            local = S.box(m, self.cfg.motion_surround, self.cfg.motion_surround)
            m = np.maximum(m - np.maximum(self.cfg.motion_floor_k * local, floor), 0.0)
        else:
            m = np.maximum(m - floor, 0.0)
        if self.cfg.motion_struct_w > 0.0 and self.ori_change is not None:
            conf = np.minimum(self.ori_change / S.MOTION_ORI_REF, 1.0)
            w = self.cfg.motion_struct_w
            m = m * ((1.0 - w) + w * conf)
        return m

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

    # --- structure maps (gradient structure-tensor, from imfeat) ---
    # Complementary to the moment grids: these see edge/gradient layout the
    # intensity moments are blind to. All on the finest grid, so (G, G).
    @property
    def struct_grid(self) -> np.ndarray:
        """(G, G, 5) per-cell structure tensor. `struct` is Optional only because a
        FrameStats can be built by hand; Gate always fills it, so the readers below go
        through here rather than each guarding the same invariant."""
        if self.struct is None:
            raise ValueError("FrameStats.struct is unset -- build it via Gate.frame()")
        return self.struct["grid_0"]

    @property
    def struct_global(self) -> np.ndarray:
        """(5,) frame-wide structure tensor. See `struct_grid`."""
        if self.struct is None:
            raise ValueError("FrameStats.struct is unset -- build it via Gate.frame()")
        return self.struct["global"]

    @property
    def edge_energy(self) -> np.ndarray:
        return self.struct_grid[:, :, S.SE_ENERGY]

    @property
    def coherence(self) -> np.ndarray:
        return self.struct_grid[:, :, S.SE_COH]  # in [0,1]; 1 = one dominant edge

    @property
    def cornerness(self) -> np.ndarray:
        return self.struct_grid[:, :, S.SE_CORN]  # Shi-Tomasi lambda_min

    @cached_property
    def orientation(self) -> np.ndarray:
        """(G,G) dominant edge orientation in radians (-pi/2, pi/2], from the
        double-angle vector; its reliability is `coherence`, kept separate."""
        g = self.struct_grid
        return 0.5 * np.arctan2(g[:, :, S.SE_OS], g[:, :, S.SE_OC])

    @property
    def sharpness(self) -> float:
        """Global gradient energy (log1p) -- a scalar detail/contrast proxy."""
        return float(np.log1p(self.struct_global[S.SE_ENERGY]))

    @cached_property
    def focus(self) -> np.ndarray:
        """(G,G) edge sharpness = gradient energy per unit intensity variance
        (~1/edge-width^2), contrast-independent: high where edges are crisp, low where
        blurred or flat -- a per-cell focus/defocus map. Unlike `sharpness` (raw energy),
        it is invariant to contrast, so it tracks focus pulls and depth-of-field, not how
        much detail or how bright the frame is."""
        return self.edge_energy / (self.v_cell_var + self.cfg.solid_thresh)

    @cached_property
    def structure_type(self) -> np.ndarray:
        """(G,G,3) soft structural decomposition [flat, edge, structured], summing to 1,
        from the per-cell structure tensor. `presence = e/(e+edge_thresh)` is how much
        gradient a cell has (0 flat .. 1 strong); among present cells, coherence splits a
        single dominant edge (high) from 2-D structure -- corners and isotropic texture,
        which share the eigenvalue signature (low). `argmax(-1)` gives a hard label.
        Corner vs texture is not separable at one scale (the tensor has only two
        eigenvalue DoF: cornerness == energy*(1-coherence)/2), so they merge here."""
        e, coh = self.edge_energy, self.coherence
        presence = e / (e + self.cfg.edge_thresh)
        return np.stack(
            [1.0 - presence, presence * coh, presence * (1.0 - coh)], axis=-1
        ).astype(np.float32)

    # --- global scene-structure descriptors (scalars from the frame-wide tensor) ---
    @property
    def orientedness(self) -> float:
        """Global edge anisotropy in [0,1]: 1 = the whole frame shares one dominant edge
        orientation (architecture, horizon), ~0 = isotropic (natural/busy scenes)."""
        return float(self.struct_global[S.SE_COH])

    @property
    def dominant_orientation(self) -> float:
        """Frame-global dominant edge orientation in radians (-pi/2, pi/2]; meaningful
        only when `orientedness` is high."""
        gv = self.struct_global
        return float(0.5 * np.arctan2(gv[S.SE_OS], gv[S.SE_OC]))

    @cached_property
    def structure_profile(self) -> np.ndarray:
        """(3,) frame-level [flat, edge, structured] fractions -- a compact scene
        signature (mean of structure_type over cells). Graphics/documents/UI skew toward
        flat+edge (clean geometry); natural photos skew structured (isotropic texture).
        """
        return self.structure_type.reshape(-1, 3).mean(0)


class FrameGate:
    """Per-frame extractor. Owns reusable buffers and one FeatureComputer; no
    temporal state, so it works identically on a still image or a video frame.
    Accepts BGR (H,W,3) or grayscale (H,W)/(H,W,1) uint8 input. Not thread-safe
    (the scratch buffers are reused per call); use one FrameGate per stream.

    Holds one imfeat worker pool of `cfg.feat_threads` threads, spawned here and
    parked between frames. One pool per FrameGate, so N streams mean N pools --
    budget against imfeat.cpu_count() if other real-time work shares the CPU."""

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        t = self.cfg.thumb
        # One extractor, one pass: moments AND structure, for every channel, on the
        # same cells. imfeat computes every feature group for every channel, always.
        self._feat = imfeat.FeatureComputer(
            shape=(t, t, 3),
            grid=[(e, e) for e in self.cfg.pyramid_exps],
            stride=self.cfg.stride,
            threads=self.cfg.feat_threads,
        )
        self._bgr = np.empty(
            (t, t, 3), np.uint8
        )  # scratch reused when not returning frames
        self._gray = np.empty((t, t), np.uint8)
        self._hsv = np.empty((t, t, 3), np.uint8)

    def _to_hsv(self, frame: np.ndarray, keep: bool) -> tuple:
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
        p = self._feat.features(hsv)
        chan = p.moments[-1].astype(np.float32)  # the 1-cell global level is last
        grids = tuple(m.astype(np.float32) for m in p.moments[: self.cfg.n_levels])
        grid = grids[
            0
        ]  # finest = output-map resolution; coarser levels feed multi-scale signals

        # Structure-tensor features, from the same pass and the same cells. imfeat
        # computes them for H, S and V; the signals below read V.
        # p.maps[i] is (H, W, C*F) with the channel axis C-major over FEATURE_NAMES,
        # so one reshape recovers (H, W, C, F) as a view. Copied, not sliced: a view
        # would retain the whole level (~400 KB) for as long as this FrameStats lives,
        # and these outlive the frame in the rolling windows. The copy is ~20 KB and
        # lands contiguous for the properties below.
        fine = p.maps[0].reshape(*p.maps[0].shape[:2], _C, _F)
        struct = {
            "grid_0": fine[:, :, S.CH_V, S.SE].copy(),
            "global": p.maps[-1].reshape(_C, _F)[S.CH_V, S.SE].copy(),
        }

        # Blank = nothing to track: flat everywhere (no cell-level intensity spread) OR
        # negligible gradient anywhere (no edge/texture energy). Both peaks come from
        # imfeat's per-level cross-cell summaries (free, same pass) -- no numpy reduction
        # here, so the cost is O(1) in grid size.
        blank = (
            float(p.summary[0][S.F_MOM + S.M_VAR, S.CH_V, S.ST_MAX])
            < self.cfg.solid_thresh
            or float(p.summary[0][S.F_SE + S.SE_ENERGY, S.CH_V, S.ST_MAX])
            < self.cfg.edge_thresh
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
            struct=struct,
            phash=int(
                p.hashes[imfeat.HASHES.index("phash"), S.CH_V]
            ),  # luma, for shot memory
        )
