"""Structure-tensor maps (from imfeat): edge energy, coherence, cornerness,
orientation, and the global sharpness scalar. These are complementary to the
moment-based maps -- they read gradient layout, not the intensity distribution.
"""

import cv2
import numpy as np
import synth

from framegate import Gate, GateConfig

G = GateConfig().grid_size


# See the note in test_robustness.py: this asserts pattern-to-cell geometry on 128px
# fixtures, so it pins the fixture-scale config rather than the 1080p-targeted default.
FIXTURE_CFG = GateConfig(thumb=256, stride=2, grid_exp=5, n_levels=4)


def test_structure_maps_shapes_and_ranges():
    fs = Gate().image(synth.checkerboard(cell=4))
    for m in (fs.edge_energy, fs.coherence, fs.cornerness, fs.orientation):
        assert m.shape == (G, G) and np.isfinite(m).all()
    assert fs.edge_energy.min() >= 0.0 and fs.cornerness.min() >= 0.0
    assert fs.coherence.min() >= 0.0 and fs.coherence.max() <= 1.0 + 1e-4
    assert isinstance(fs.sharpness, float)


def test_edge_energy_tracks_detail():
    busy = Gate().image(synth.checkerboard(cell=4)).edge_energy.mean()
    ramp = Gate().image(synth.gradient()).edge_energy.mean()
    flat = Gate().image(synth.solid(128)).edge_energy.mean()
    assert busy > ramp > flat
    assert flat < 1e-3  # a solid field has no gradient


def test_cornerness_peaks_on_corners():
    corners = Gate().image(synth.checkerboard(cell=4)).cornerness.max()
    ramp = Gate().image(synth.gradient()).cornerness.max()  # edges but no corners
    assert corners > 5.0 * (ramp + 1e-6)


def test_orientation_separates_horizontal_from_vertical():
    # gradient along x -> vertical-ish edges; along y -> horizontal edges. The
    # double-angle orientation should differ markedly between the two.
    ox = Gate().image(synth.gradient(axis=1)).orientation.mean()
    oy = Gate().image(synth.gradient(axis=0)).orientation.mean()
    assert abs(ox - oy) > 0.5


def test_sharpness_drops_under_blur():
    img = synth.checkerboard(cell=4)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    assert Gate().image(img).sharpness > Gate().image(blur).sharpness


def test_structure_maps_cached_and_present_on_video():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    g = Gate()
    fs, _ = g.frame(img)
    assert fs.orientation is fs.orientation  # cached_property
    assert fs.struct["global"].shape == (5,)


def test_focus_map_localizes_blur_and_is_contrast_invariant():
    scene = synth.hsv_scene(60, 2)
    w = scene.shape[1]
    half = scene.copy()
    half[:, : w // 2] = cv2.GaussianBlur(scene, (0, 0), 3)[:, : w // 2]
    f = Gate().image(half).focus
    assert f[:, 16:].mean() > 3.0 * f[:, :16].mean()  # sharp half >> blurred half
    # contrast-invariant: halving global contrast barely moves focus (energy ~var both drop)
    low = np.clip((scene.astype(np.float32) - 128) * 0.5 + 128, 0, 255).astype(np.uint8)
    full, dim = Gate().image(scene).focus.mean(), Gate().image(low).focus.mean()
    assert 0.5 < dim / full < 2.0


def test_focus_decreases_monotonically_with_blur():
    base = synth.hsv_scene(60, 2)
    means = [
        Gate().image(base if s == 0 else cv2.GaussianBlur(base, (0, 0), s)).focus.mean()
        for s in (0.0, 1.0, 2.0, 4.0)
    ]
    assert all(a > b for a, b in zip(means, means[1:]))  # strictly decreasing


def test_structure_type_soft_labels():
    g = Gate(FIXTURE_CFG)
    edge = np.full((256, 256, 3), 40, np.uint8)
    edge[:, 128:] = 210
    for img in (synth.solid(128), edge, synth.checkerboard(cell=4), synth.foliage()):
        st = g.image(img).structure_type
        assert st.shape[-1] == 3 and np.allclose(st.sum(-1), 1.0, atol=1e-4)
        assert st.min() >= 0.0
    assert g.image(synth.solid(128)).structure_type[..., 0].mean() > 0.95  # ~all flat
    # checkerboard / foliage are 2-D structure (low coherence) -> 'structured' dominates
    assert g.image(synth.checkerboard(cell=4)).structure_profile.argmax() == 2
    assert g.image(synth.foliage()).structure_profile.argmax() == 2


def test_orientedness_separates_graphic_from_natural():
    g = Gate()
    # coherent man-made structure (rules, single edge) -> high orientedness
    edge = np.full((256, 256, 3), 40, np.uint8)
    edge[:, 128:] = 210
    assert g.image(synth.rules()).orientedness > 0.8
    assert g.image(edge).orientedness > 0.8
    # isotropic natural texture -> low orientedness
    assert g.image(synth.foliage()).orientedness < 0.3
    assert g.image(synth.checkerboard(cell=4)).orientedness < 0.3


def test_dominant_orientation_flips_with_structure():
    g = Gate()
    horiz = synth.stripes(period=6, horizontal=True)  # varies along x -> vertical edges
    vert = synth.stripes(period=6, horizontal=False)
    assert (
        abs(g.image(horiz).dominant_orientation - g.image(vert).dominant_orientation)
        > 0.5
    )
