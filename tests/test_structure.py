"""Structure-tensor maps (from structstats): edge energy, coherence, cornerness,
orientation, and the global sharpness scalar. These are complementary to the
moment-based maps -- they read gradient layout, not the intensity distribution.
"""

import cv2
import numpy as np

from framegate import Gate, GateConfig
import synth

G = GateConfig().grid_size


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
