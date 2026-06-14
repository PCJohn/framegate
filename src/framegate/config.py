"""Configuration for the gate.

All tunables live in one immutable dataclass -- the single source of truth. Build one:

    GateConfig()                                  # library defaults
    GateConfig(min_scene_len=6, thumb=96)         # override in code
    GateConfig.from_yaml("my.yaml")               # load from a file
    GateConfig.from_yaml("my.yaml", thumb=96)     # file + code overrides
    GateConfig.from_yaml()                        # = library defaults (reads no file)

There is no shipped YAML to drift from the dataclass: ``from_yaml()`` with no path just
uses the defaults, and ``to_yaml()`` generates a template from the live fields on demand
(``python -m framegate`` prints one).
"""

from dataclasses import dataclass, fields, replace

import yaml


@dataclass(frozen=True)
class GateConfig:
    # --- frame extraction ---
    thumb: int = 256  # thumbnail side for stats + FAST
    stride: int = (
        2  # tensorstats grid stride; >1 = indexed-gather over subsampled pixels
    )
    fast_thumb: int = 128  # resolution for the FAST blank-check (subsampled from thumb)
    grid_exp: int = (
        5  # 2^grid_exp cells per dim (5 -> 32x32); finest / output-map level
    )
    n_levels: int = (
        4  # dyadic pyramid levels, coarsening from grid_exp (5,4,3,2 -> 32..4)
    )
    fast_thresh: int = 10  # FAST corner threshold (tier-2 blank check)
    solid_thresh: float = 1.0  # blank if max V cell-variance < this

    # --- cut score ---
    shift_search: int = 3  # motion-compensation radius in cells
    cut_w_luma: float = 1.0  # weight of the luma-structure path
    cut_w_color: float = 1.0  # weight of the global colour-shift path
    ncc_flattol: float = 1.0  # luma std below this -> no structure, skip luma path
    color_maxd: float = (
        2.0  # max normalized chroma-vector distance (scales colour path)
    )
    fast_static: bool = (
        True  # skip the shift search when zero-shift corr is already this high
    )
    static_corr: float = (
        0.98  # (effectively lossless: only fires when motion can't change the cut)
    )

    # --- cut decision ---
    roll_win: int = 20  # rolling window for robust (median+MAD) scoring
    robust_min: int = 8  # samples before the rolling score is trusted
    robust_k: float = 8.0  # outlier if value > median + k * 1.4826 * MAD
    cut_dissim: float = 0.45  # static-scene guard floor on the cut score
    min_scene_len: int = 6  # min frames between cuts (debounce)
    freeze_eps: float = 0.20  # residual-RMS + |dV| below this -> frozen frame

    # --- free temporal signals ---
    flicker_win: int = 32  # brightness-history length (also flicker FFT length)
    fade_win: int = 8  # frames over which a fade ramp is measured
    fade_span: float = 60.0  # V-mean change treated as a full fade

    # --- fine-texture map ---
    ftex_achromatic_w: float = 0.5  # down-weight saturated cells (achromatic prior)
    ftex_coarse_k: int = 3  # neighborhood for the coarse between-cell energy
    ftex_line_k: int = 5  # horizontal smoothing window (text-line coherence)

    # --- saliency map ---
    sal_surround: int = 7  # neighborhood (cells) for the center-surround luma contrast

    # --- motion map ---
    motion_floor_k: float = (
        1.0  # subtract k * local-mean|residual| as the noise floor (0 = off)
    )
    motion_surround: int = 7  # neighborhood (cells) for that local floor
    motion_abs_floor: float = (
        1.0  # also subtract at least this many grey levels (0 = off); both 0 = raw |residual|
    )

    # --- output ---
    return_frames: bool = (
        True  # attach the thumbnail + HSV to FrameStats for the caller to reuse
    )

    # --- video-level optimization ---
    skip_duplicates: bool = True  # reuse stats for byte-identical consecutive frames

    @property
    def grid_size(self) -> int:
        return 2**self.grid_exp

    @property
    def pyramid_exps(self) -> list:
        """Per-level spatial cell-exponents, finest->coarsest: [grid_exp .. grid_exp-n_levels+1].
        Level 0 is the finest (output-map) grid; coarser levels feed multi-scale signals.
        """
        exps = [self.grid_exp - i for i in range(self.n_levels)]
        if exps[-1] < 0:
            raise ValueError(
                f"n_levels={self.n_levels} too deep for grid_exp={self.grid_exp}"
            )
        return exps

    @classmethod
    def from_yaml(cls, path=None, **overrides) -> "GateConfig":
        """Build from a YAML file plus optional code overrides. With no path, returns the
        library defaults (overrides still apply) -- no file is read."""
        if path is None:
            data = {}
        else:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        data = {**data, **overrides}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def to_yaml(cls) -> str:
        """A YAML template generated from the live dataclass defaults -- the single source
        of truth, so it cannot drift. Copy, edit, and load with from_yaml(path)."""
        head = (
            "# framegate config template (generated from GateConfig defaults).\n"
            "# See the GateConfig dataclass for what each key does.\n\n"
        )
        lines = [
            f"{f.name}: {str(f.default).lower() if isinstance(f.default, bool) else f.default}"
            for f in fields(cls)
        ]
        return head + "\n".join(lines) + "\n"

    def replace(self, **overrides) -> "GateConfig":
        return replace(self, **overrides)
