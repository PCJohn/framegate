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

from dataclasses import MISSING, dataclass, field, fields, replace

import imfeat
import yaml

RESIZE_INTERP = ("area", "nearest")  # thumbnail filters FrameGate knows


def _yaml_val(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, dict):
        return yaml.safe_dump(v, default_flow_style=True).strip()
    return v


def _default_of(f):
    return f.default if f.default is not MISSING else f.default_factory()


@dataclass(frozen=True)
class GateConfig:
    # --- frame extraction ---
    # The thumbnail, its resize filter and the stride are fastdet's front-end exactly
    # (Detector.front_end_spec), so one imfeat pass can feed both the gate and a fastdet
    # model. The cheaper, coarser operating point is stride=4, resize_interp="nearest".
    # thumb is a side in pixels (a square every frame is resized to, smaller ones
    # upscaled) or an imfeat policy name that sizes it from the frame, never an upscale.
    # All start from the square of the shorter side's power of two: "pow2" is that square
    # (720p -> 512, 1080p -> 1024, 4K -> 2048); "pow2-fit" keeps the frame's shape inside
    # it, the longer side the power of two (720p -> 512x320, 1080p -> 1024x576, 4K ->
    # 2048x1152), never more pixels than the square and cells a power of two wide, which
    # imfeat's pass is fastest at -- the cheapest rule; "pow2-cover" keeps the shape
    # around the square (720p -> 896x512), the dearest. The aspect policies keep the
    # shape only as closely as the grid allows: the scaled side is rounded to a multiple
    # of 64, so 16:9 is exact at widths 1024 and 2048 but 1.6:1 at 512, and a banner too
    # thin for that rounding gets the square. All are floored at the grid (a frame under
    # 64 px is upscaled to it as a fixed size would). thumb_hw(shape) gives a frame's
    # size; the stride is capped at the cell side there so no cell of a small frame goes
    # unsampled.
    thumb: int | str = "pow2-fit"  # thumbnail for stats (made inside imfeat's pass)
    resize_interp: str = (
        "area"  # thumbnail filter: "area" (box, like fastdet) or "nearest" (cv2)
    )
    stride: int = 1  # grid stride; >1 = indexed-gather over a subsample
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
    fade_win: int = 8  # frames over which a fade ramp is measured (brightness history)
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

    # --- learned maps (fastdet models; see models.py) ---
    # <name>.fdt files in models_dir (relative to the package) load by name, and models
    # maps a name to a path on top (None drops a bundled one). Each runs on the gate's
    # imfeat pass and lands on FrameStats.model_maps[name]; a "text" model replaces the
    # heuristic text map. fastdet is optional: without it, bundled models are skipped.
    models_dir: str = "models"
    models: dict = field(default_factory=dict)
    # Threads the fastdet scorer uses per frame (its own pool, apart from imfeat's); 0 =
    # feat_threads. The map is bit-identical at any count; only the time changes, and
    # the scorer splits its work by tiles, so it scales where imfeat's pass has stopped.
    model_threads: int = 0

    # --- output ---
    # attach the thumbnail to FrameStats for caller reuse (its HSV, made on demand)
    return_frames: bool = True

    # --- video-level optimization ---
    skip_duplicates: bool = True  # reuse stats for byte-identical consecutive frames

    def __post_init__(self):
        if self.resize_interp not in RESIZE_INTERP:
            raise ValueError(
                f"resize_interp={self.resize_interp!r}: want one of {RESIZE_INTERP}"
            )
        if isinstance(self.thumb, str):
            if self.thumb not in imfeat.THUMB_POLICIES:
                raise ValueError(
                    f"thumb={self.thumb!r}: want a side in pixels or one of "
                    f"{imfeat.THUMB_POLICIES}"
                )
        elif (
            isinstance(self.thumb, bool)
            or not isinstance(self.thumb, int)
            or self.thumb < 1
        ):
            raise ValueError(f"thumb={self.thumb!r}: want a side in pixels or a policy")

    @property
    def grid_size(self) -> int:
        return 2**self.grid_exp

    def thumb_hw(self, shape: tuple) -> tuple:
        """The (rows, cols) thumbnail a frame of `shape` ((H, W) or (H, W, C)) gets: the
        fixed square, or the policy's size floored at the grid. fastdet's
        features.thumb_hw is the same rule."""
        rows, cols = imfeat.thumb_size(shape[:2], self.thumb)
        if isinstance(self.thumb, str):
            g = self.grid_size
            rows, cols = max(rows, g), max(cols, g)
        return rows, cols

    def stride_for(self, size: tuple) -> int:
        """The stride on a thumbnail of `size`: cfg.stride, at most the cell side (imfeat
        restarts the stride at cell edges along columns, not rows, so a wider stride would
        leave cells without a sample). fastdet applies the same rule (features.stride_for),
        so a model's features match."""
        return max(1, min(self.stride, min(size) // self.grid_size))

    def samples_per_cell_for(self, shape: tuple) -> int:
        """Sampled pixels per finest cell, per dimension (the shorter one, off a square),
        on a frame of `shape`: what the stride costs. Four is the practical floor for the
        moment and histogram features to carry information; the default (stride 1) has 9
        on a 1080p frame's 1024x576 thumbnail, 5 on 720p's 512x320, 18 on 4K's 2048x1152.
        """
        size = self.thumb_hw(shape)
        return (min(size) // self.grid_size) // self.stride_for(size)

    @property
    def samples_per_cell(self) -> int:
        """`samples_per_cell_for` of a frame the thumbnail is a fixed square of (with a
        policy the thumbnail follows the frame, so ask `samples_per_cell_for` about one;
        this is then the 1080p frame's). Below 1 the stride steps clean over whole cells
        and some end up with no samples at all -- imfeat reports a count of 0 there and
        every derived feature is meaningless -- which `pyramid_exps` refuses for a fixed
        square and `stride_for` caps away under a policy."""
        if isinstance(self.thumb, str):
            return self.samples_per_cell_for((1080, 1920))
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
        if isinstance(self.thumb, int) and self.samples_per_cell < 1:
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
        lines = [f"{f.name}: {_yaml_val(_default_of(f))}" for f in fields(cls)]
        return head + "\n".join(lines) + "\n"

    def replace(self, **overrides) -> "GateConfig":
        return replace(self, **overrides)
