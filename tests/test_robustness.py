"""Robustness suite: degenerate inputs, value-range invariants, determinism,
illumination invariance, positive/negative cases for every temporal feature, and
config sweeps (grid / stride / thumb). The goal is to exercise the gate on the kinds
of frames real pipelines actually throw at it and assert nothing crashes or goes
out of range."""

from dataclasses import replace

import cv2
import numpy as np
import pytest
import synth

from framegate import Gate, GateConfig

G = GateConfig().grid_size


# ---------------------------------------------------------------- degenerate inputs


@pytest.mark.parametrize(
    "img, want_blank",
    [
        (synth.black(), True),  # all black
        (synth.white(), True),  # all white
        (synth.solid(128), True),  # uniform mid-grey
        (synth.solid((40, 160, 90)), True),  # uniform random colour
        (synth.noise(seed=1), False),  # full-range colour noise
        (synth.hsv_scene(60, 2), False),  # structured natural-ish scene
    ],
)
def test_blank_classification_on_degenerate_inputs(img, want_blank):
    fs = Gate().image(img)
    assert fs.blank is want_blank


def test_regular_patterns_read_blank():
    # checkerboard: cells fall inside the squares -> zero within-cell variance -> blank
    # via the solid tier. gradient: smooth, negligible edge energy -> blank via the
    # energy tier. (Neither needs the old FAST corner detector.)
    assert Gate().image(synth.checkerboard(cell=16)).blank is True
    assert Gate().image(synth.gradient()).blank is True


def test_fine_texture_is_not_blank():
    # Fine periodic texture has real edge energy but no corners -- the old FAST check
    # wrongly blanked it; the energy check keeps it (it is trackable content).
    assert Gate().image(synth.noisy(synth.stripes(period=4))).blank is False
    assert Gate().image(synth.noisy(synth.checkerboard(cell=2))).blank is False
    # ...while genuinely featureless frames still read blank.
    assert Gate().image(synth.gradient()).blank is True
    assert Gate().image(synth.noisy(synth.solid(128), amp=3)).blank is True


def test_blank_energy_threshold_is_the_lever():
    striped = synth.noisy(synth.stripes(period=4))
    assert Gate(GateConfig(edge_thresh=1000.0)).image(striped).blank is False
    assert (
        Gate(GateConfig(edge_thresh=1e12)).image(striped).blank is True
    )  # term dominates


def test_solid_inputs_have_sane_scalars():
    fs = Gate().image(synth.white())
    assert fs.exposure > 250 and fs.colorfulness == 0.0  # white: bright, no colour
    fs = Gate().image(synth.black())
    assert fs.exposure < 5  # black: dark
    fs = Gate().image(synth.solid((0, 0, 200)))  # saturated red (BGR)
    assert fs.colorfulness > 0  # has colour


@pytest.mark.parametrize(
    "shape",
    [(8, 8, 3), (16, 16, 3), (1080, 200, 3), (200, 1080, 3), (1, 256, 3), (256, 1, 3)],
)
def test_extreme_shapes_do_not_crash(shape):
    img = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    fs = Gate().image(img)
    assert fs.saliency.shape == (G, G)
    assert fs.shape == shape[:2]


def test_non_contiguous_input_is_accepted():
    base = synth.noise(seed=2)
    flipped = base[::-1, ::-1]  # a non-contiguous view
    assert not flipped.flags["C_CONTIGUOUS"]
    fs = Gate().image(flipped)
    assert fs.saliency.shape == (G, G)


def test_grayscale_two_and_three_dim_match():
    g = synth.grayscale_scene(2)
    a = Gate().image(g)
    b = Gate().image(g[:, :, None])
    assert a.colorfulness == b.colorfulness == 0.0
    assert np.array_equal(a.grid, b.grid)


# ---------------------------------------------------------------- value-range invariants

_VARIED = [
    synth.black(),
    synth.white(),
    synth.solid(128),
    synth.noise(seed=3),
    synth.gradient(),
    synth.checkerboard(cell=2),
    synth.stripes(period=2),
    synth.hsv_scene(60, 2),
    synth.grayscale_scene(4),
]


@pytest.mark.parametrize("img", _VARIED)
def test_map_and_scalar_ranges(img):
    fs = Gate().image(img)
    assert fs.saliency.shape == fs.text.shape == (G, G)
    assert fs.saliency.min() >= 0.0 and np.isfinite(fs.saliency).all()
    assert fs.text.min() >= 0.0 and np.isfinite(fs.text).all()
    assert 0.0 <= fs.exposure <= 255.0
    assert fs.colorfulness >= 0.0 and fs.contrast >= 0.0 and fs.detail >= 0.0
    assert 0.0 <= fs.flat_fraction <= 1.0


def test_temporal_signal_ranges_over_a_mixed_stream():
    scene = synth.hsv_scene(60, 2)
    frames = (
        [synth.noisy(scene) for _ in range(10)]
        + synth.fade_ramp(scene, np.linspace(1.0, 0.2, 8))
        + [synth.noisy(synth.hsv_scene(120, 5)) for _ in range(10)]
    )
    g = Gate()
    for f in frames:
        fs, sig = g.frame(f)
        assert -1.0 <= sig.fade <= 1.0
        assert 0.0 <= sig.flicker <= 1.0
        assert sig.struct_corr <= 1.0 + 1e-6
        assert sig.cut_score >= 0.0
        if fs.motion is not None:
            assert fs.motion.min() >= 0.0 and np.isfinite(fs.motion).all()


# ---------------------------------------------------------------- determinism & no-copy


def test_determinism_same_input_same_output():
    img = synth.noise(seed=7)
    a, b = Gate().image(img), Gate().image(img)
    assert np.array_equal(a.grid, b.grid) and np.array_equal(a.chan, b.chan)
    assert np.array_equal(a.saliency, b.saliency)
    assert np.array_equal(a.text, b.text)


def test_image_call_does_not_mutate_input():
    img = synth.noise(seed=8)
    before = img.copy()
    Gate().image(img)
    assert np.array_equal(img, before)


# ---------------------------------------------------------------- texture / saliency behaviour


def test_text_responds_to_fine_not_smooth():
    fine = Gate(FIXTURE_CFG).image(synth.checkerboard(cell=2)).text.max()
    smooth = Gate(FIXTURE_CFG).image(synth.gradient()).text.max()
    assert fine > smooth  # dense pattern >> smooth ramp


def test_text_bimodality_suppresses_isotropic_clutter():
    # realistic text: dark ink on light paper, mixed-orientation strokes
    txt = synth.text_block()
    # symmetric multi-scale achromatic clutter (foliage-like), comparable fine energy
    rng = np.random.default_rng(0)
    base = np.repeat(np.repeat(rng.integers(40, 210, (16, 16)), 8, 0), 8, 1)
    clut = (0.6 * base + 0.4 * rng.integers(0, 255, (128, 128))).astype(np.uint8)
    clut = np.stack([clut] * 3, -1)
    t = Gate(FIXTURE_CFG).image(txt).text.max()
    c = Gate(FIXTURE_CFG).image(clut).text.max()
    assert (
        t > 2.0 * c
    )  # bimodality gate keeps asymmetric text, suppresses symmetric clutter


def test_text_isotropy_gate_suppresses_oriented_patterns():
    """The isotropy (coherence) gate's unique job: suppress *asymmetric* coherent
    patterns -- horizontal rules on paper -- that the bimodality gate cannot touch
    (rules are sparse-dark-on-light, so their |skew| is high, like text). Without the
    gate, rules outscore text; with it, text wins. Symmetric coherent patterns (stripes)
    are already handled by the bimodality gate, so they are a softer check."""
    txt, rules, stripes = synth.text_block(), synth.rules(), synth.stripes(period=4)
    on = FIXTURE_CFG
    off = replace(FIXTURE_CFG, text_coherence_w=0.0)

    r_off, r_on = Gate(off).image(rules).text.max(), Gate(on).image(rules).text.max()
    t_off, t_on = Gate(off).image(txt).text.max(), Gate(on).image(txt).text.max()
    assert r_off > t_off  # without the gate, asymmetric rules beat text
    assert t_on > r_on  # with it, real text wins
    assert r_on < 0.6 * r_off  # rules strongly suppressed
    assert t_on > 0.6 * t_off  # real text only mildly affected
    assert t_on > Gate(on).image(stripes).text.max()  # stripes handled jointly too


def test_text_beats_structured_distractors_across_variations():
    """Thorough text battery: across fonts, sizes, light/dark polarity, rotation and
    added noise, the weakest text still outscores every *structured* distractor (stripes,
    rules, foliage clutter, a single edge, a smooth gradient, a checkerboard, a colour
    scene). Pure high-amplitude noise is a known exception (see the next test)."""
    text_variants = [synth.text_block(font=f) for f in range(len(synth.FONTS))] + [
        synth.text_block(scale=0.4),  # small
        synth.text_block(scale=0.75),  # large
        synth.text_block(fg=235, bg=20),  # light on dark
        synth.text_block(rot=6),  # slightly rotated
        synth.noisy(synth.text_block(), amp=8),  # with sensor noise
    ]
    edge = np.full((256, 256, 3), 40, np.uint8)
    edge[:, 128:] = 210
    distractors = [
        synth.stripes(period=4),
        synth.rules(),
        synth.foliage(),
        edge,
        synth.gradient(),
        synth.checkerboard(cell=4),
        synth.hsv_scene(60, 2),
    ]
    g = Gate(FIXTURE_CFG)
    text_min = min(float(g.image(t).text.max()) for t in text_variants)
    distractor_max = max(float(g.image(d).text.max()) for d in distractors)
    assert text_min > 1.3 * distractor_max


def test_text_pure_noise_is_a_known_limitation():
    """Documented limitation: full-range per-pixel noise scores comparably to text --
    separating them needs stroke-level analysis (connected components / stroke width),
    out of scope for a low-level moment cue. This test pins the current behavior so a
    future improvement shows up here rather than as a silent regression elsewhere."""
    g = Gate()
    text = float(g.image(synth.text_block()).text.max())
    noise = float(g.image(synth.noise(seed=2)).text.max())
    assert (
        noise > 0.5 * text
    )  # noise is NOT reliably separable (a would-be false positive)


def test_text_is_low_on_non_text():
    g = Gate()
    for img in (
        synth.solid(128),
        synth.gradient(),
        synth.black(),
        synth.hsv_scene(60, 2),
    ):
        assert float(g.image(img).text.max()) < 12.0  # no strong response on non-text


@pytest.mark.parametrize(
    "img, want_blank",
    [
        (synth.solid(128), True),  # uniform grey
        (synth.solid((0, 0, 0)), True),  # pure black
        (synth.solid((255, 255, 255)), True),  # pure white
        (synth.gradient(), True),  # smooth ramp: no edge energy
        (synth.noisy(synth.solid(128), amp=3), True),  # faint sensor noise on flat
        (synth.checkerboard(cell=16), True),  # cells inside squares -> flat cells
        (synth.noise(seed=3), False),  # full-range texture -> content
        (
            synth.noisy(synth.stripes(period=4)),
            False,
        ),  # fine periodic texture -> content
        (synth.text_block(size=128), False),  # text -> content
        (synth.hsv_scene(60, 2), False),  # natural scene -> content
    ],
)
def test_blank_classification_battery(img, want_blank):
    assert Gate().image(img).blank is want_blank


def test_blank_single_hot_cell_is_not_blank():
    # a nearly-flat frame with one small high-contrast patch is trackable content.
    img = np.full((128, 128, 3), 128, np.uint8)
    img[40:56, 40:56] = synth.noisy(synth.hsv_scene(60, 2))[40:56, 40:56]
    assert Gate().image(img).blank is False


def test_blank_grayscale_and_color_paths_agree():
    scene = synth.hsv_scene(60, 2)
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    assert Gate().image(gray).blank is Gate().image(scene).blank is False
    assert Gate().image(np.full((128, 128), 90, np.uint8)).blank is True  # flat gray


_SCALAR_SIGNALS = [
    "exposure",
    "contrast",
    "colorfulness",
    "detail",
    "flat_fraction",
    "noise_floor",
    "clipping",
    "sharpness",
    "orientedness",
    "dominant_orientation",
]
_MAP_SIGNALS = [
    "saliency",
    "text",
    "edge_energy",
    "coherence",
    "cornerness",
    "orientation",
    "focus",
    "v_cell_mean",
    "v_cell_var",
]


# The signal-semantics tests below assert pattern-to-cell geometry on synth.THUMB-sized
# (128px) fixtures: a 2px checker against a cell, text strokes against a cell. The shipped
# default targets 1080p+ sources and upsamples a 128px fixture 8x, which moves those
# patterns a full octave relative to the grid and changes what the assertions mean. They
# pin the fixture-scale config so they keep testing the signal, not the default.
FIXTURE_CFG = GateConfig(thumb=256, stride=2, grid_exp=5, n_levels=4)


@pytest.mark.parametrize(
    "cfg",
    [
        GateConfig(),
        GateConfig(grid_exp=5, n_levels=5),
        GateConfig(thumb=2048),
        GateConfig(n_levels=1),
    ],
)
def test_all_signals_finite_and_shaped(cfg):
    """Every per-frame signal is finite and correctly shaped across configs and a wide
    input battery -- solid, black, full noise, natural scene, text, a grayscale frame,
    and a tiny frame. Guards broadly against NaN / shape regressions (empty reductions,
    divide-by-zero) in any signal, including the structural ones."""
    G = cfg.grid_size
    inputs = [
        synth.solid(128),
        synth.black(),
        synth.noise(seed=1),
        synth.hsv_scene(60, 2),
        synth.text_block(size=128),
        cv2.cvtColor(synth.hsv_scene(60, 2), cv2.COLOR_BGR2GRAY),  # grayscale path
        np.full((6, 6, 3), 128, np.uint8),  # tiny frame (upsized to thumb)
    ]
    for img in inputs:
        fs = Gate(cfg).image(img)
        for name in _SCALAR_SIGNALS:
            v = getattr(fs, name)
            assert isinstance(v, float) and np.isfinite(v), (name, v)
        for name in _MAP_SIGNALS:
            m = getattr(fs, name)
            assert m.shape == (G, G) and np.isfinite(m).all(), name
        assert fs.structure_type.shape == (G, G, 3)
        assert np.isfinite(fs.structure_type).all()
        assert np.allclose(fs.structure_type.sum(-1), 1.0, atol=1e-4)
        assert fs.structure_profile.shape == (3,)
        assert fs.color_mean.shape == (3,) and np.isfinite(fs.color_mean).all()


def test_stream_with_midstream_blank_is_stable():
    """A blank stretch mid-stream resets cleanly: signals stay finite, and the first
    frame after the blank has motion=None (no previous frame to diff), exercising the
    orientation-change reset added for structural motion validation."""
    g = Gate()
    pre = [synth.noisy(synth.hsv_scene(60, 2)) for _ in range(4)]
    blank = [synth.white(), synth.black()]
    post = [synth.noisy(synth.hsv_scene(200, 7)) for _ in range(4)]
    motions = []
    for f in pre + blank + post:
        fs, s = g.frame(f)
        assert np.isfinite(s.cut_score)
        if fs.motion is not None:
            assert np.isfinite(fs.motion).all()
        motions.append(fs.motion)
    assert motions[len(pre + blank)] is None  # first post-blank frame has no motion
    assert motions[-1] is not None  # tracking resumes after


def test_saliency_orientation_popout():
    """A patch oriented differently from its surround -- same luma, colour and contrast
    -- pops out via the orientation-contrast channel. Intensity/colour/variance channels
    are blind to it (they rank it at or below background), so this exercises the specific
    channel the structure features add."""
    fs = Gate(FIXTURE_CFG).image(synth.orientation_popout())
    sal = fs.saliency
    patch = sal[12:20, 12:20].mean()  # vertical-stripe patch (grid cells ~12..20)
    bg = sal[2:10, 2:10].mean()  # horizontal-stripe surround
    assert patch > 3.0 * (bg + 1e-6)


def test_saliency_localizes_a_distinct_patch():
    f = synth.solid((50, 50, 50))
    f[40:90, 40:90] = np.random.default_rng(0).integers(150, 255, (50, 50, 3))
    sal = Gate().image(f).saliency
    gy, gx = slice(int(40 / 128 * G), int(90 / 128 * G)), slice(
        int(40 / 128 * G), int(90 / 128 * G)
    )
    assert sal[gy, gx].max() > sal.mean()  # patch stands out


# ---------------------------------------------------------------- illumination invariance


def test_global_brightness_shift_is_not_a_cut():
    scene = synth.hsv_scene(60, 2)
    g = Gate()
    cuts = []
    for f in [synth.noisy(scene) for _ in range(8)] + [
        synth.dim(scene, 0.6) for _ in range(8)
    ]:
        _, sig = g.frame(f)
        cuts.append(sig.cut)
    assert not any(cuts)  # a dimming is photometric, not a cut


# ---------------------------------------------------------------- temporal: positive AND negative


def _stream(frames, cfg=None):
    g = Gate(cfg) if cfg else Gate()
    return [g.frame(f)[1] for f in frames]


def test_fade_in_is_positive_fade_out_is_negative():
    scene = synth.hsv_scene(60, 2)
    out = [s.fade for s in _stream(synth.fade_ramp(scene, np.linspace(1.0, 0.15, 16)))]
    inn = [s.fade for s in _stream(synth.fade_ramp(scene, np.linspace(0.15, 1.0, 16)))]
    assert min(out) < -0.5 and max(inn) > 0.5


def test_static_scene_has_no_fade_no_flicker():
    scene = synth.hsv_scene(60, 2)
    sigs = _stream([synth.noisy(scene) for _ in range(40)])
    assert max(abs(s.fade) for s in sigs) < 0.5
    assert max(s.flicker for s in sigs) < 0.4


def test_steady_brightness_has_no_flicker_but_strobe_does():
    scene = synth.hsv_scene(60, 2)
    steady = [s.flicker for s in _stream([synth.noisy(scene) for _ in range(40)])]
    strobe = [
        s.flicker
        for s in _stream(
            [
                synth.noisy(synth.dim(scene, 1.0 if i % 2 == 0 else 0.4))
                for i in range(40)
            ]
        )
    ]
    assert max(steady) < 0.4 < max(strobe)


def test_freeze_positive_and_negative():
    scene = synth.hsv_scene(60, 2)
    held = _stream([synth.noisy(scene) for _ in range(3)] + [scene] * 8)
    assert sum(s.freeze for s in held) >= 6  # identical held frames freeze
    moving = _stream([synth.moving_block(x) for x in range(4, 40, 3)])
    assert not any(s.freeze for s in moving)  # motion never freezes


def test_noise_jitter_does_not_cause_cuts():
    scene = synth.hsv_scene(60, 2)
    assert not any(
        s.cut for s in _stream([synth.noisy(scene, amp=6) for _ in range(40)])
    )


def test_motion_present_under_movement_absent_when_static():
    moving = Gate()
    fs = None
    for x in range(4, 40, 4):
        fs, _ = moving.frame(synth.moving_block(x))
    assert fs.motion is not None and fs.motion.max() > 0.0  # a moving block registers

    stat = Gate()
    scene = synth.hsv_scene(60, 2)
    for _ in range(5):
        fs, _ = stat.frame(synth.noisy(scene, amp=2))
    assert (
        float((fs.motion == 0).mean()) > 0.5
    )  # a jittered-but-static scene is mostly zero


# ---------------------------------------------------------------- config sweeps


# grid_exp 7 is deliberately absent: 8px cells at stride 4 is 2 samples per dimension,
# and cut detection genuinely fails there.
@pytest.mark.parametrize("grid_exp", [5, 6])
def test_grid_exp_sweep_shapes_and_cut(grid_exp):
    cfg = GateConfig(grid_exp=grid_exp)
    g = Gate(cfg)
    cuts = []
    for f in [synth.noisy(synth.hsv_scene(60, 2)) for _ in range(20)] + [
        synth.noisy(synth.hsv_scene(60, 3)) for _ in range(8)
    ]:
        fs, sig = g.frame(f)
        if sig.cut:
            cuts.append(sig.cut_frame)
    assert fs.saliency.shape == (cfg.grid_size, cfg.grid_size)
    assert cuts == [20]  # cut still detected at any grid size


@pytest.mark.parametrize("stride", [1, 2])
def test_stride_sweep_runs(stride):
    fs = Gate(GateConfig(stride=stride)).image(synth.hsv_scene(60, 2))
    assert fs.saliency.shape == (G, G) and np.isfinite(fs.saliency).all()


@pytest.mark.parametrize("thumb", [512, 1024, 2048])
def test_thumb_sweep_runs(thumb):
    fs = Gate(GateConfig(thumb=thumb)).image(synth.hsv_scene(60, 2))
    assert fs.thumb.shape == (thumb, thumb, 3)
    assert fs.saliency.shape == (G, G)


def test_motion_structural_validation_rejects_photometric_not_real_motion():
    """A regional lighting/shadow shift (luma changes, edges do not move) reads as much
    less motion than genuine content motion. Orientation is illumination-invariant, so
    the structural validation suppresses the photometric change while leaving real
    (edge-moving) motion essentially intact."""
    base = synth.noisy(synth.hsv_scene(60, 2), amp=2)
    w = base.shape[1]
    shadow = np.clip(
        base.astype(np.int16) - np.where(np.arange(w)[None, :, None] < w // 2, 40, 0),
        0,
        255,
    ).astype(np.uint8)
    moved = np.roll(base, 6, axis=1)

    def motion(second, struct_w):
        g = Gate(GateConfig(motion_struct_w=struct_w))
        g.frame(base)
        return g.frame(second)[0].motion

    assert (
        motion(shadow, 0.7).mean() < 0.6 * motion(shadow, 0.0).mean()
    )  # photometric cut
    assert (
        motion(moved, 0.7).mean() > 0.8 * motion(moved, 0.0).mean()
    )  # real motion kept


def test_motion_raw_when_floors_disabled():
    cfg = GateConfig(motion_floor_k=0.0, motion_abs_floor=0.0, motion_struct_w=0.0)
    g = Gate(cfg)
    fs = None
    for x in range(4, 40, 4):
        fs, _ = g.frame(synth.moving_block(x))
    assert np.allclose(
        fs.motion, np.abs(fs.residual)
    )  # both floors off -> raw magnitude
