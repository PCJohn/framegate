"""Configuration for the gate.

All tunables live in one immutable dataclass. Three ways to build one:

    GateConfig()                                  # library defaults
    GateConfig(min_scene_len=6, thumb=96)         # override in code
    GateConfig.from_yaml("my.yaml")               # load from a file
    GateConfig.from_yaml("my.yaml", thumb=96)     # file + code overrides
    GateConfig.from_yaml()                        # the packaged default.yaml

The shipped ``default.yaml`` mirrors these defaults (a test enforces that), so it
doubles as a documented template to copy and edit.
"""

from dataclasses import dataclass, fields, replace
from importlib import resources

import yaml


@dataclass(frozen=True)
class GateConfig:
    # --- frame extraction ---
    thumb: int = 128          # thumbnail side for stats + FAST
    stride: int = 1           # tensorstats stride (1 hits the uint8 fast path)
    grid_exp: int = 5         # 2^grid_exp cells per dim (5 -> 32x32)
    fast_thresh: int = 10     # FAST corner threshold (tier-2 blank check)
    solid_thresh: float = 1.0  # blank if max V cell-variance < this

    # --- cut score ---
    shift_search: int = 3     # motion-compensation radius in cells
    cut_w_luma: float = 1.0   # weight of the luma-structure path
    cut_w_color: float = 1.0  # weight of the global colour-shift path
    ncc_flattol: float = 1.0  # luma std below this -> no structure, skip luma path
    color_maxd: float = 2.0   # max normalized chroma-vector distance (scales colour path)
    fast_static: bool = True  # skip the shift search when zero-shift corr is already this high
    static_corr: float = 0.98  # (effectively lossless: only fires when motion can't change the cut)

    # --- cut decision ---
    roll_win: int = 20        # rolling window for robust (median+MAD) scoring
    robust_min: int = 8       # samples before the rolling score is trusted
    robust_k: float = 8.0     # outlier if value > median + k * 1.4826 * MAD
    cut_dissim: float = 0.45  # static-scene guard floor on the cut score
    min_scene_len: int = 6    # min frames between cuts (debounce)
    freeze_eps: float = 0.20  # residual-RMS + |dV| below this -> frozen frame

    # --- free temporal signals ---
    flicker_win: int = 32     # brightness-history length (also flicker FFT length)
    fade_win: int = 8         # frames over which a fade ramp is measured
    fade_span: float = 60.0   # V-mean change treated as a full fade

    # --- fine-texture map ---
    ftex_achromatic_w: float = 0.5  # down-weight saturated cells (achromatic prior)
    ftex_coarse_k: int = 3          # neighborhood for the coarse between-cell energy
    ftex_line_k: int = 5            # horizontal smoothing window (text-line coherence)

    # --- roi ---
    roi_k: float = 1.0         # a cell is active if it deviates > this many std from its map's mean
    roi_merge_iou: float = 0.75  # merge proposal boxes whose IoU exceeds this (lower = more aggressive)

    # --- output ---
    return_frames: bool = True    # attach the thumbnail + HSV to FrameStats for the caller to reuse

    # --- video-level optimization ---
    skip_duplicates: bool = True  # reuse stats for byte-identical consecutive frames

    @property
    def grid_size(self) -> int:
        return 2 ** self.grid_exp

    @classmethod
    def from_yaml(cls, path=None, **overrides) -> "GateConfig":
        if path is None:
            text = resources.files("framegate").joinpath("default.yaml").read_text()
        else:
            with open(path) as f:
                text = f.read()
        data = {**(yaml.safe_load(text) or {}), **overrides}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def replace(self, **overrides) -> "GateConfig":
        return replace(self, **overrides)
