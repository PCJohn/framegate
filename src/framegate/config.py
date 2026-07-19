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


def _yaml_val(v):
    return str(v).lower() if isinstance(v, bool) else v


@dataclass(frozen=True)
class GateConfig:
    # --- frame extraction ---
    thumb: int = 256  # thumbnail side for stats
    stride: int = 2  # grid stride; >1 = indexed-gather over a subsample
    grid_exp: int = 5  # 2^grid_exp cells/dim (5 -> 32x32); output/finest level
    n_levels: int = 4  # dyadic pyramid levels from grid_exp (5,4,3,2 -> 32..4)
    solid_thresh: float = 1.0  # blank if max V cell-variance < this
    edge_thresh: float = 1000.0  # ...or if max cell edge-energy < this

    # --- cut score ---
    shift_search: int = 3  # motion-compensation radius in cells
    ncc_flattol: float = 1.0  # luma std below this -> no structure, skip luma path
    color_maxd: float = 2.0  # max normalized chroma-vector distance (colour path)
    fast_static: bool = True  # skip shift search when zero-shift corr already high
    static_corr: float = 0.98  # lossless: only fires when motion can't change the cut

    # --- cut decision ---
    roll_win: int = 20  # rolling window for robust (median+MAD) scoring
    robust_min: int = 8  # samples before the rolling score is trusted
    robust_k: float = 8.0  # outlier if value > median + k * 1.4826 * MAD
    cut_dissim: float = 0.45  # static-scene guard floor on the cut score
    reid_z: float = 6.0  # shot re-ID: match a stored shot group if MAD-z <= this
    min_scene_len: int = 6  # min frames between cuts (debounce)
    freeze_eps: float = 0.20  # residual-RMS + |dV| below this -> frozen frame

    # --- free temporal signals ---
    flicker_win: int = 32  # brightness-history length (also flicker FFT length)
    fade_win: int = 8  # frames over which a fade ramp is measured
    fade_span: float = 60.0  # V-mean change treated as a full fade

    # --- text map ---
    text_achromatic_w: float = 0.5  # down-weight saturated cells (achromatic prior)
    text_coarse_k: int = 3  # neighborhood for the coarse between-cell energy
    text_line_k: int = 5  # horizontal smoothing window (text-line coherence)
    text_skew_w: float = 0.8  # bimodality gate: suppress symmetric clutter, 0=off
    text_skew_ref: float = 1.2  # |standardized skew| at which the gate saturates
    text_coherence_w: float = 0.8  # isotropy gate: suppress oriented edges, 0=off

    # --- saliency map ---
    sal_surround: int = 7  # neighborhood (cells) for the center-surround luma contrast

    # --- motion map ---
    motion_floor_k: float = 1.0  # k*local-mean|residual| noise floor (0 = off)
    motion_surround: int = 7  # neighborhood (cells) for that local floor
    motion_abs_floor: float = 1.0  # absolute grey-level floor (0 = off)
    motion_struct_w: float = (
        0.7  # down-weight luma change with no edge motion (0 = off)
    )

    # --- output ---
    return_frames: bool = True  # attach thumb+HSV to FrameStats for caller reuse

    # --- video-level optimization ---
    skip_duplicates: bool = True  # reuse stats for byte-identical consecutive frames

    @property
    def grid_size(self) -> int:
        return 2**self.grid_exp

    @property
    def pyramid_exps(self) -> list:
        """Per-level cell-exponents, finest->coarsest: [grid_exp .. grid_exp-n_levels].
        Level 0 is the finest (output) grid; coarser levels feed multi-scale signals.
        """
        exps = [self.grid_exp - i for i in range(self.n_levels)]
        if exps[-1] < 0:
            raise ValueError(
                f"n_levels={self.n_levels} too deep for grid_exp={self.grid_exp}"
            )
        return exps

    @classmethod
    def from_yaml(cls, path=None, **overrides) -> "GateConfig":
        """Build from a YAML file plus optional code overrides. With no path, return the
        library defaults (overrides still apply) -- no file is read."""
        if path is None:
            data: dict = {}
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
        """A YAML template generated from the live dataclass defaults -- one source
        of truth, so it cannot drift. Copy, edit, and load with from_yaml(path)."""
        head = (
            "# framegate config template (generated from GateConfig defaults).\n"
            "# See the GateConfig dataclass for what each key does.\n\n"
        )
        lines = [f"{f.name}: {_yaml_val(f.default)}" for f in fields(cls)]
        return head + "\n".join(lines) + "\n"

    def replace(self, **overrides) -> "GateConfig":
        return replace(self, **overrides)
