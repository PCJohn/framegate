"""Configuration for the gate.

All tunables live in one immutable dataclass -- the single source of truth. Build one:

    GateConfig()                                  # library defaults
    GateConfig(min_scene_len=6, thumb=512)        # override in code
    GateConfig.from_yaml("my.yaml")               # load from a file
    GateConfig.from_yaml("my.yaml", thumb=512)    # file + code overrides
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
    thumb: int = 1024  # thumbnail side for stats
    stride: int = 4  # grid stride; >1 = indexed-gather over a subsample
    feat_threads: int = 2  # imfeat worker threads; output is bit-identical at any count
    grid_exp: int = 6  # 2^grid_exp cells/dim (6 -> 64x64); output/finest level
    n_levels: int = 6  # dyadic pyramid levels from grid_exp (6..1 -> 64..2)
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
    # L2 shot re-ID (framestore + per-bit Bernoulli). A new shot's first-frame pHash
    # queries the store for candidate groups within reid_maxd (relative Hamming, the
    # recall net); each candidate is then scored by the log-likelihood RATIO, in nats,
    # that the frame came from that shot's learned bit distribution rather than from
    # the population of setups seen so far. A group scores as the MIXTURE over its
    # prototypes (weighted by the share of the group's shot openings each holds), so a
    # group spread over a pan pays ~log K rather than getting K attempts at the bar;
    # those ratios become a posterior over "which group, or a new one" under a
    # Chinese-restaurant prior of concentration reid_alpha, and the winner is a re-ID
    # iff its posterior log-odds >= reid_llr (the precision dial). Log-odds are in the
    # same nats as the bare ratio and a lone candidate with one shot at reid_alpha = 1
    # scores exactly its ratio, so the prior only acts where there is something to weigh
    # against: a close rival splits the posterior and blocks both, more known groups
    # raise the bar, and a setup that has already recurred is likelier to recur again. Bits that vary within a shot (a moving mouth)
    # stop penalising the match, and bits that every setup in the footage shares stop
    # supporting it, so what remains is evidence that is both stable and distinctive.
    # Scale: an agreeing informative bit is worth ~0.7 nats and a disagreeing one costs
    # ~3.9, so reid_llr = 8 asks for roughly a dozen distinctive bits in agreement.
    reid_maxd: float = 0.25  # candidate radius: relative Hamming in [0,1]
    reid_llr: float = 8.0  # match if the posterior log-odds (nats) >= this
    reid_alpha: float = 1.0  # CRP concentration: prior weight on "a new setup"
    reid_eps: float = (
        0.02  # per-bit flip rate the model always allows for (see reid.py)
    )
    min_scene_len: int = 6  # min frames between cuts (debounce)
    # L1 memory: a frame is frozen if it affine-matches any of the last freeze_win kept
    # frames. freeze_win=1 compares against the previous frame only (two-frame L1, the
    # default); a larger window lets a flicker between two held frames still read frozen.
    freeze_eps: float = 0.275  # residual-RMS + |dV| below this -> frozen frame
    freeze_win: int = 1  # L1 ring: recent kept frames to compare against

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
    def samples_per_cell(self) -> int:
        """Sampled pixels per finest cell, per dimension. Below 1 the stride steps clean
        over whole cells and some end up with no samples at all -- imfeat reports a count
        of 0 there and every derived feature is meaningless. Four is the practical floor
        for the moment and histogram features to carry information; the default sits
        exactly on it."""
        return (self.thumb // self.grid_size) // self.stride

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
        if self.samples_per_cell < 1:
            raise ValueError(
                f"thumb={self.thumb} over a {self.grid_size}x{self.grid_size} grid gives "
                f"{self.thumb // self.grid_size}px cells, which stride={self.stride} steps "
                f"over entirely -- some cells would get no samples"
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
