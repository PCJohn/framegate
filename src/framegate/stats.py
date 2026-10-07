"""Per-frame extraction: FrameStats (the stateless descriptor) and FrameGate
(the extractor that produces it). Works on a single image or a video frame; for
video the temporal layer (StreamAnalyzer) consumes a stream of FrameStats.
"""

from dataclasses import dataclass, field
from functools import cached_property

import cv2
import imfeat
import numpy as np

from . import signals as S
from .config import GateConfig
from .models import ModelBank

# The gate hands imfeat the BGR thumbnail and lets it convert to HSV inside its pass
# (FeatureComputer's input_space / feature_space, which came with imfeat.COLOR_SPACES).
# An imfeat without that would take the thumbnail as HSV and every signal would be
# wrong, so it is refused here rather than found in the maps.
if not hasattr(imfeat, "COLOR_SPACES"):
    raise ImportError(
        "framegate needs an imfeat that converts BGR to HSV inside its pass "
        "(pip install git+https://github.com/PCJohn/imfeat)"
    )

_F = len(imfeat.FEATURE_NAMES)  # 38 features per channel in a pyramid map
_C = 3  # HSV
_INTERP = {"area": cv2.INTER_AREA, "nearest": cv2.INTER_NEAREST}  # cfg.resize_interp


_TMP: dict = {}


def _tmp(shape: tuple) -> np.ndarray:
    """A float32 array of `shape` for a computation's intermediates, kept and reused:
    the gate runs one frame at a time, and nothing is kept past the computation."""
    a = _TMP.get(shape)
    if a is None:
        a = _TMP[shape] = np.empty(shape, np.float32)
    return a


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
    cell_means: np.ndarray | None = (
        None  # (C, G, G) == grid[:, :, :, M_MEAN] channel-major: each channel's map of
    )
    #   cell means contiguous, for the per-frame readers (the temporal layer); a view of
    #   grid has a 48-byte column stride, which costs three to four times per operation
    thumb: np.ndarray | None = (
        None  # the input at thumbnail size (BGR or gray), if cfg.return_frames
    )
    residual: np.ndarray | None = (
        None  # (G,G) photometric change vs the previous frame;
    )
    #   set by StreamAnalyzer, None for a standalone image or the first/post-blank frame
    prev_ovec: np.ndarray | None = (
        None  # the previous frame's (G,G,2) edge-orientation vectors, for ori_change;
    )
    #   set by StreamAnalyzer with residual, None when residual is
    struct: dict | None = (
        None  # imfeat structure maps for V: "grid_0" (cells,cells,5) + "global" (5,)
    )
    phash: int = 0  # whole-frame luma pHash (imfeat), one uint64; shot-memory key
    model_maps: dict = field(
        default_factory=dict
    )  # name -> (G,G) float32 probabilities from the fastdet models loaded (models.py)

    @cached_property
    def hsv(self) -> np.ndarray | None:
        """The thumbnail in HSV, if cfg.return_frames: what the signals were computed on.

        Made on first access, from `thumb`: imfeat converts inside the gate's pass now,
        so there is no HSV image to hand back unless asked for. `imfeat.convert` gives
        cv2.cvtColor's bytes; a grayscale thumbnail becomes H = S = 0, V = luma."""
        if self.thumb is None:
            return None
        if self.thumb.ndim == 2:
            hsv = np.zeros((*self.thumb.shape, 3), np.uint8)
            hsv[:, :, 2] = self.thumb
            return hsv
        return imfeat.convert(self.thumb)

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

    def _cell_mean(self, ch: int) -> np.ndarray:
        """grid[:, :, ch, M_MEAN], contiguous: from cell_means when the gate made it."""
        if self.cell_means is not None:
            return self.cell_means[ch]
        return np.ascontiguousarray(self.grid[:, :, ch, S.M_MEAN])

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
        """(G,G) text likelihood. With a ``text`` model loaded (models.py) this is its
        probability map in [0, 1]; otherwise the heuristic cue from low-level texture --
        fine, achromatic, coherent, bimodal -- an unnormalised score tuned for
        dense/printed text (body text, captions, UI). A cue, not OCR. See signals.text.
        """
        learned = self.model_maps.get("text")
        if learned is not None:
            return learned
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
        averaged over cells. Unsaturated cells (hue = noise) contribute ~nothing.

        The three means are [sat.mean(), (sat * cos(ang)).mean(), (sat * sin(ang)).mean()]
        with ang = hue * (pi / 90) (OpenCV hue 0..180 -> 0..2pi), from the contiguous
        planes: the two products are taken in one call into a shared scratch array and
        summed in one call -- numpy sums each row in the order it sums the array on its
        own -- and each sum is divided by the count as a mean is; the values are those
        three means to the bit, for half the calls."""
        sat, hue = self._cell_mean(S.CH_S), self._cell_mean(S.CH_H)
        n = sat.size
        trig = _tmp((2, *sat.shape))
        ang = hue * (np.pi / 90.0)
        np.cos(ang, out=trig[0])
        np.sin(ang, out=trig[1])
        hv = np.multiply(sat, trig, out=trig).reshape(2, -1).sum(axis=1)
        return np.array([sat.sum() / n, hv[0] / n, hv[1] / n], np.float32)

    @cached_property
    def ori_change(self) -> np.ndarray | None:
        """(G,G) ||d(edge orientation vector)|| vs the previous frame; illumination-
        invariant (orientation does not change when lighting does). None for a standalone
        image or the first/post-blank frame. Computed when first read: only `motion`
        (with motion_struct_w > 0) uses it, so a frame whose motion map nobody reads
        never pays for it."""
        if self.prev_ovec is None:
            return None
        ovec = self.struct_grid[:, :, S.SE_OC : S.SE_OS + 1]
        return np.hypot(
            ovec[:, :, 0] - self.prev_ovec[:, :, 0],
            ovec[:, :, 1] - self.prev_ovec[:, :, 1],
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
    """Per-frame extractor. Owns reusable buffers and the FeatureComputers; no
    temporal state, so it works identically on a still image or a video frame.
    Accepts BGR (H,W,3) or grayscale (H,W)/(H,W,1) uint8 input. Not thread-safe
    (the scratch buffers are reused per call); use one FrameGate per stream.

    Holds imfeat worker pools of `cfg.feat_threads` threads, spawned here and parked
    between frames, and the fastdet models the config names (models.py), which score
    the same pass. A pool per FrameGate (two when both kinds of frame arrive, see
    `process`), so N streams mean N pools -- budget against imfeat.cpu_count() if other
    real-time work shares the CPU. `close()` joins the pools; call it before interpreter
    shutdown in long-running hosts."""

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self._models = ModelBank(self.cfg)
        # One pass: moments AND structure, for every channel, on the same cells. imfeat
        # computes every feature group for every channel, always, converts the BGR
        # thumbnail to HSV as it reads it (cv2.cvtColor's bytes, at a fraction of its
        # cost, with no second image) and, given the frame, makes the thumbnail itself
        # inside the pass (cv2.resize(INTER_AREA)'s bytes, likewise). The thumbnail's
        # size follows the frame under a policy (cfg.thumb_hw), so the computers are kept
        # by size, all built on first use: `_fused` takes a frame of the size last seen
        # and resizes it inside its pass; `_feat` holds one per thumbnail size for ready
        # thumbnails, the frames cv2 has to resize (see _fuses) -- a stream has one size,
        # a set of images a few.
        self._fused: imfeat.FeatureComputer | None = None
        self._fused_shape: tuple | None = None
        self._feat: dict[tuple, imfeat.FeatureComputer] = {}
        self._closed = False
        # scratch for the cv2 path, reused when not returning frames; sized on first use
        self._bgr: np.ndarray | None = None
        self._gray: np.ndarray | None = None

    def _computer(
        self, shape: tuple, size: tuple, thumb: bool = False
    ) -> imfeat.FeatureComputer:
        """The pass for a thumbnail of `size` -- made inside it from a frame of `shape`
        with `thumb`, else taken ready (`shape` is then the thumbnail's)."""
        return imfeat.FeatureComputer(
            shape=shape,
            grid=[(e, e) for e in self.cfg.pyramid_exps],
            stride=self.cfg.stride_for(size),
            threads=self.cfg.feat_threads,
            input_space="bgr",
            feature_space="hsv",
            thumb=size if thumb else None,
        )

    def _fuses(self, frame: np.ndarray) -> bool:
        """Whether imfeat thumbnails this frame inside its pass: a BGR frame at least the
        thumbnail's size in both axes (under a policy, every frame of 64 px or more), with
        the "area" filter (imfeat's resize is INTER_AREA, downscaling only). The rest --
        grayscale, smaller frames (an upscale is bilinear in OpenCV), "nearest" -- go
        through cv2.resize and the thumbnail computer, with the same numbers out."""
        if not (
            self.cfg.resize_interp == "area" and frame.ndim == 3 and frame.shape[2] == 3
        ):
            return False
        rows, cols = self.cfg.thumb_hw(frame.shape)
        return frame.shape[0] >= rows and frame.shape[1] >= cols

    def _scratch(self, size: tuple) -> tuple:
        """The cv2 path's reused `(bgr, gray)` buffers, remade when the size changes."""
        if self._bgr is None or self._bgr.shape[:2] != size:
            self._bgr = np.empty((*size, 3), np.uint8)
            self._gray = np.empty(size, np.uint8)
        return self._bgr, self._gray

    def _thumbnail(self, frame: np.ndarray, size: tuple, keep: bool) -> tuple:
        """The cv2 path (see _fuses): resize to the thumbnail (cfg.resize_interp):
        `(bgr, thumb)`, the (rows, cols, 3) array imfeat converts and reads, and the
        thumbnail to hand back (BGR, or the grayscale itself). A grayscale frame is
        replicated into the three channels, so imfeat sees H=S=0, V=luma and colour
        signals correctly read as zero. With `keep`, `thumb` is a fresh array the caller
        can hold; otherwise a reused scratch buffer.
        """
        rows, cols = size
        interp = _INTERP[self.cfg.resize_interp]
        bgr, gray = self._scratch(size)
        if frame.ndim == 2 or frame.shape[2] == 1:
            thumb = np.empty((rows, cols), np.uint8) if keep else gray
            cv2.resize(
                frame.reshape(frame.shape[0], frame.shape[1]),
                (cols, rows),
                dst=thumb,
                interpolation=interp,
            )
            cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR, dst=bgr)
            return bgr, (thumb if keep else None)
        thumb = np.empty((rows, cols, 3), np.uint8) if keep else bgr
        cv2.resize(frame, (cols, rows), dst=thumb, interpolation=interp)
        return thumb, (thumb if keep else None)

    def process(self, frame: np.ndarray) -> FrameStats:
        if self._closed:
            raise RuntimeError("this FrameGate is closed")
        h, w = frame.shape[:2]
        keep = self.cfg.return_frames
        size = self.cfg.thumb_hw(frame.shape)
        if self._fuses(frame):
            if self._fused is None or self._fused_shape != frame.shape:
                self._fused = self._computer(
                    frame.shape, size, thumb=True
                )  # a new frame size
                self._fused_shape = frame.shape
            thumb = np.empty((*size, 3), np.uint8) if keep else None
            fused: imfeat.FeatureComputer = self._fused
            p = fused.features(frame, thumb_out=thumb)
        else:
            feat = self._feat.get(size)
            if feat is None:
                feat = self._feat[size] = self._computer((*size, 3), size)
            bgr, thumb = self._thumbnail(frame, size, keep)
            p = feat.features(bgr)
        # The learned maps first: they consume the imfeat result, which is not kept (a
        # FrameStats outlives its frame in the rolling windows, and the result is
        # megabytes), and a model in the config is a request to run it. First, because
        # they walk megabytes of level maps and would push the grids below out of the
        # cache; made after them, the grids are still in cache when the temporal layer
        # reads their planes a moment later (a 48-byte column stride touches every line
        # of a grid, so a cold plane costs the whole grid's worth of misses).
        model_maps = self._models.maps(p, (h, w)) if self._models else {}

        chan = p.moments[-1].astype(np.float32)  # the 1-cell global level is last
        grids = tuple(m.astype(np.float32) for m in p.moments[: self.cfg.n_levels])
        grid = grids[
            0
        ]  # finest = output-map resolution; coarser levels feed multi-scale signals
        cell_means = np.ascontiguousarray(grid[:, :, :, S.M_MEAN].transpose(2, 0, 1))

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
            cell_means=cell_means,
            blank=blank,
            shape=(h, w),
            cfg=self.cfg,
            thumb=thumb,
            struct=struct,
            phash=int(
                p.hashes[imfeat.HASHES.index("phash"), S.CH_V]
            ),  # luma, for shot memory
            model_maps=model_maps,
        )

    @property
    def models(self) -> list[str]:
        """Names of the loaded models (the keys of `FrameStats.model_maps`)."""
        return self._models.names

    @property
    def model_threads(self) -> dict[str, int]:
        """Threads each model's scorer runs on (cfg.model_threads, or feat_threads)."""
        return self._models.threads if self._models else {}

    def close(self) -> None:
        """Join the imfeat pools and the models' scorer threads. The gate is unusable
        afterwards. Python does this when the object dies, but do it explicitly before
        interpreter shutdown on Windows, where joining threads during DLL unload can
        stall the process."""
        self._models.close()
        self._feat = {}
        self._fused = None
        self._closed = True
