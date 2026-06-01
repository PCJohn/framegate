"""Accuracy tests. These read like a user driving the public API."""

import numpy as np

from framegate import Gate, GateConfig
import synth


def _cuts(frames, cfg=None):
    g = Gate(cfg) if cfg else Gate()
    out = []
    for f in frames:
        _, sig = g.frame(f)
        if sig.cut:
            out.append(sig.cut_frame)
    return out


def test_normal_cut_fires_at_boundary():
    a = [synth.noisy(synth.hsv_scene(60, 2)) for _ in range(20)]
    b = [synth.noisy(synth.hsv_scene(60, 3)) for _ in range(8)]
    assert _cuts(a + b) == [20]


def test_equiluminant_colour_cut_is_caught():
    # identical luma layout (same vseed), different hue -> only the colour path sees it
    a = [synth.noisy(synth.hsv_scene(15, 1)) for _ in range(20)]
    b = [synth.noisy(synth.hsv_scene(120, 1)) for _ in range(8)]
    assert _cuts(a + b) == [20]


def test_pan_produces_no_false_cut():
    rng = np.random.default_rng(1)
    big = rng.random((300, 500))
    import cv2
    big = cv2.GaussianBlur(big, (0, 0), 1.2)
    big = np.clip((big - big.mean()) * 1.6 + big.mean(), 0, 1)
    bigc = cv2.cvtColor((big * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pan = [synth.noisy(cv2.resize(bigc[10:138, x:x + 128], (128, 128))) for x in range(0, 60, 2)]
    assert _cuts(pan) == []


def test_blank_and_white_flash_are_blank_not_cut():
    c = synth.hsv_scene(60, 2)
    white = np.full((128, 128, 3), 255, np.uint8)
    g = Gate()
    flags = []
    for f in [c] * 5 + [synth.black(), white, synth.black()] + [c] * 5:
        fs, sig = g.frame(f)
        flags.append((fs.blank, sig.cut))
    assert flags[5] == (True, False)   # black -> blank, not cut
    assert flags[6] == (True, False)   # white flash -> blank, not cut


def test_freeze_on_held_frame():
    c = synth.hsv_scene(60, 2)
    _, freezes, _, _ = synth.run_stream(Gate(), [synth.noisy(c) for _ in range(3)] + [c] * 8)
    assert freezes >= 6


def test_min_scene_len_debounce():
    c, d, e = synth.hsv_scene(60, 2), synth.hsv_scene(60, 3), synth.hsv_scene(60, 7)
    frames = ([synth.noisy(c) for _ in range(20)] +
              [synth.noisy(d) for _ in range(4)] +    # cut at 20, next cut 4 frames later
              [synth.noisy(e) for _ in range(10)])
    assert _cuts(frames, GateConfig(min_scene_len=6)) == [20]            # second suppressed
    assert len(_cuts(frames, GateConfig(min_scene_len=2))) == 2          # both allowed


def test_fade_is_signed_and_not_a_cut():
    c = synth.hsv_scene(60, 2)
    frames = [synth.noisy(synth.dim(c, f)) for f in np.linspace(1.0, 0.15, 16)]
    cuts, _, fades, _ = synth.run_stream(Gate(), frames)
    assert cuts == []
    assert min(fades) < -0.5     # strong darkening


def test_flicker_detected_without_false_cuts():
    c = synth.hsv_scene(60, 2)
    frames = [synth.noisy(synth.dim(c, 1.0 if i % 2 == 0 else 0.45)) for i in range(40)]
    cuts, _, _, flickers = synth.run_stream(Gate(), frames)
    assert cuts == []
    assert max(flickers) > 0.4


def test_grayscale_input_is_colorless_but_detects_cuts():
    g2 = synth.grayscale_scene(2)
    g3 = synth.grayscale_scene(3)
    assert Gate().image(g2).colorfulness == 0.0          # no colour in grayscale
    assert Gate().image(g2[:, :, None]).colorfulness == 0.0   # (H,W,1) also accepted
    cuts = _cuts([synth.noisy(g2) for _ in range(20)] + [synth.noisy(g3) for _ in range(8)])
    assert cuts == [20]


def test_rois_trim_to_subject_and_drop_full_frame():
    rng = np.random.default_rng(2)
    frame = np.full((400, 600, 3), 20, np.uint8)
    frame[140:260, 220:380] = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    rois = Gate().image(frame).rois
    assert rois                                       # something localized
    (x0, y0, x1, y1), labels = rois[0]                # consensus region hugs the subject
    assert 180 < x0 < 240 and 360 < x1 < 420
    assert 110 < y0 < 170 and 240 < y1 < 290
    assert len(labels) >= 2 and all(isinstance(s, str) for s in labels)
    busy = (rng.random((400, 600, 3)) * 255).astype(np.uint8)
    assert Gate().image(busy).rois == []              # nothing localizes -> empty (use whole frame)


def test_single_image_path_needs_no_stream():
    fs = Gate().image(synth.hsv_scene(60, 2))
    for k in ["exposure", "contrast", "colorfulness", "detail", "clipping", "noise_floor"]:
        assert isinstance(getattr(fs, k), float)
    assert fs.saliency.shape == (GateConfig().grid_size,) * 2


def test_duplicate_skip_is_lossless():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    seq = [img, img.copy(), rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)]

    def outputs(skip):
        g = Gate(GateConfig(skip_duplicates=skip))
        res = []
        for f in seq:
            fs, sig = g.frame(f)
            res.append((sig.cut, round(sig.cut_score, 6), sig.freeze, round(sig.struct_corr, 6)))
        return res

    assert outputs(True) == outputs(False)


def test_rois_are_labelled_and_localized():
    rng = np.random.default_rng(2)
    frame = np.full((400, 600, 3), 20, np.uint8)
    frame[140:260, 220:380] = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    rois = Gate().image(frame).rois
    assert rois
    area = 400 * 600
    seen = set()
    for (x0, y0, x1, y1), labels in rois:
        assert (x1 - x0) * (y1 - y0) < 0.95 * area               # every box localizes
        assert labels and labels == sorted(labels)               # labelled, deterministic order
        assert (x0, y0, x1, y1) not in seen                      # no duplicate boxes
        seen.add((x0, y0, x1, y1))
    g = synth.grayscale_scene(2)
    assert isinstance(Gate().image(g).rois, list)                # grayscale path runs


def test_rois_merge_aggregates_labels():
    rng = np.random.default_rng(2)
    frame = np.full((400, 600, 3), 20, np.uint8)
    frame[140:260, 220:380] = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    fs = Gate().image(frame)
    aggressive = fs.rois                                         # default merge_iou = 0.5
    none = Gate(GateConfig(roi_merge_iou=1.0)).image(frame).rois  # only exact duplicates removed
    assert len(aggressive) <= len(none)                          # merging reduces the count
    assert sum(len(l) for _, l in aggressive) >= 8               # all map proposals accounted for


def test_return_frames_default_on_and_toggleable():
    t = GateConfig().thumb
    fs = Gate().image(synth.hsv_scene(60, 2))
    assert fs.thumb.shape == (t, t, 3) and fs.hsv.shape == (t, t, 3)
    g = Gate().image(synth.grayscale_scene(2))
    assert g.thumb.shape == (t, t) and g.hsv.shape == (t, t, 3)   # grayscale thumb is 1-channel
    off = Gate(GateConfig(return_frames=False)).image(synth.hsv_scene(60, 2))
    assert off.thumb is None and off.hsv is None


def test_fast_static_matches_full_search_on_cuts():
    a = [synth.noisy(synth.hsv_scene(60, 2)) for _ in range(20)]
    b = [synth.noisy(synth.hsv_scene(60, 3)) for _ in range(8)]
    on = _cuts(a + b, GateConfig(fast_static=True))
    off = _cuts(a + b, GateConfig(fast_static=False))
    assert on == off == [20]


def test_appearance_maps_cached_motion_aware_rois_recompute():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    g = Gate()
    fs1, _ = g.frame(img)
    assert fs1.saliency is fs1.saliency                      # appearance maps cached (pure per-frame)
    assert fs1.fine_texture is fs1.fine_texture
    fs2, _ = g.frame(img.copy())                             # byte-identical -> stats reused
    assert fs2 is fs1                                         # duplicate-skip returns the same object
    assert fs2.saliency is fs1.saliency                      # appearance reused for free
    # rois is recomputed (depends on motion, which is not a pure function of one frame)
    assert isinstance(fs2.rois, list)


def test_motion_roi_only_on_video_with_movement():
    # a bright block that jumps location each frame -> a moving region
    def frame_at(x):
        f = np.full((240, 420, 3), 30, np.uint8)
        f[60:180, x:x + 120] = 220
        return f
    g = Gate()
    labels_seen = set()
    for x in (40, 70, 100, 130, 160, 190):
        fs, _ = g.frame(frame_at(x))
        for _, labels in fs.rois:
            labels_seen.update(labels)
    assert "motion" in labels_seen                           # video path proposes a motion ROI
    # a standalone image never has motion
    img_labels = {l for _, ls in Gate().image(frame_at(100)).rois for l in ls}
    assert "motion" not in img_labels
