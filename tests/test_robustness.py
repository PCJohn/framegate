"""Robustness suite: degenerate inputs, value-range invariants, determinism,
illumination invariance, positive/negative cases for every temporal feature, and
config sweeps (grid / stride / thumb). The goal is to exercise the gate on the kinds
of frames real pipelines actually throw at it and assert nothing crashes or goes
out of range."""

import numpy as np
import pytest

from framegate import Gate, GateConfig
import synth

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


def test_regular_patterns_without_fast_corners_read_blank():
    # a clean checkerboard / gradient has high variance but no FAST corner arcs,
    # so the gate (correctly) treats it as having nothing localizable to detect.
    assert Gate().image(synth.checkerboard(cell=16)).blank is True
    assert Gate().image(synth.gradient()).blank is True


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
    fine = Gate().image(synth.checkerboard(cell=2)).text.max()
    smooth = Gate().image(synth.gradient()).text.max()
    assert fine > smooth  # dense pattern >> smooth ramp


def test_text_bimodality_suppresses_isotropic_clutter():
    # asymmetric "text": sparse dark marks on light paper, in horizontal rows
    rng = np.random.default_rng(0)
    txt = np.full((128, 128, 3), 225, np.uint8)
    for r in range(8, 122, 10):
        for c in range(6, 122, 2):
            if rng.random() < 0.55:
                txt[r : r + 5, c : c + 1] = 30
    # symmetric multi-scale achromatic clutter (foliage-like), comparable fine energy
    base = np.repeat(np.repeat(rng.integers(40, 210, (16, 16)), 8, 0), 8, 1)
    clut = (0.6 * base + 0.4 * rng.integers(0, 255, (128, 128))).astype(np.uint8)
    clut = np.stack([clut] * 3, -1)
    t = Gate().image(txt).text.max()
    c = Gate().image(clut).text.max()
    assert (
        t > 3.0 * c
    )  # bimodality gate keeps asymmetric text, suppresses symmetric clutter


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


@pytest.mark.parametrize("grid_exp", [4, 5, 6])
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


@pytest.mark.parametrize("thumb", [64, 96, 128, 160])
def test_thumb_sweep_runs(thumb):
    fs = Gate(GateConfig(thumb=thumb)).image(synth.hsv_scene(60, 2))
    assert fs.thumb.shape == (thumb, thumb, 3)
    assert fs.saliency.shape == (G, G)


def test_motion_raw_when_floors_disabled():
    cfg = GateConfig(motion_floor_k=0.0, motion_abs_floor=0.0)
    g = Gate(cfg)
    fs = None
    for x in range(4, 40, 4):
        fs, _ = g.frame(synth.moving_block(x))
    assert np.allclose(
        fs.motion, np.abs(fs.residual)
    )  # both floors off -> raw magnitude
