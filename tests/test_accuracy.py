"""Accuracy tests. These read like a user driving the public API."""

import numpy as np
import synth

from framegate import Gate, GateConfig


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
    pan = [
        synth.noisy(cv2.resize(bigc[10:138, x : x + 128], (128, 128)))
        for x in range(0, 60, 2)
    ]
    assert _cuts(pan) == []


def test_blank_and_white_flash_are_blank_not_cut():
    c = synth.hsv_scene(60, 2)
    white = np.full((128, 128, 3), 255, np.uint8)
    g = Gate()
    flags = []
    for f in [c] * 5 + [synth.black(), white, synth.black()] + [c] * 5:
        fs, sig = g.frame(f)
        flags.append((fs.blank, sig.cut))
    assert flags[5] == (True, False)  # black -> blank, not cut
    assert flags[6] == (True, False)  # white flash -> blank, not cut


def test_freeze_on_held_frame():
    c = synth.hsv_scene(60, 2)
    _, freezes, _, _ = synth.run_stream(
        Gate(), [synth.noisy(c) for _ in range(3)] + [c] * 8
    )
    assert freezes >= 6


def test_min_scene_len_debounce():
    c, d, e = synth.hsv_scene(60, 2), synth.hsv_scene(60, 3), synth.hsv_scene(60, 7)
    frames = (
        [synth.noisy(c) for _ in range(20)]
        + [synth.noisy(d) for _ in range(4)]  # cut at 20, next cut 4 frames later
        + [synth.noisy(e) for _ in range(10)]
    )
    assert _cuts(frames, GateConfig(min_scene_len=6)) == [20]  # second suppressed
    assert len(_cuts(frames, GateConfig(min_scene_len=2))) == 2  # both allowed


def test_fade_is_signed_and_not_a_cut():
    c = synth.hsv_scene(60, 2)
    frames = [synth.noisy(synth.dim(c, f)) for f in np.linspace(1.0, 0.15, 16)]
    cuts, _, fades, _ = synth.run_stream(Gate(), frames)
    assert cuts == []
    assert min(fades) < -0.5  # strong darkening


def test_flicker_detected_without_false_cuts():
    c = synth.hsv_scene(60, 2)
    frames = [synth.noisy(synth.dim(c, 1.0 if i % 2 == 0 else 0.45)) for i in range(40)]
    cuts, _, _, flickers = synth.run_stream(Gate(), frames)
    assert cuts == []
    assert max(flickers) > 0.4


def test_grayscale_input_is_colorless_but_detects_cuts():
    g2 = synth.grayscale_scene(2)
    g3 = synth.grayscale_scene(3)
    assert Gate().image(g2).colorfulness == 0.0  # no colour in grayscale
    assert Gate().image(g2[:, :, None]).colorfulness == 0.0  # (H,W,1) also accepted
    cuts = _cuts(
        [synth.noisy(g2) for _ in range(20)] + [synth.noisy(g3) for _ in range(8)]
    )
    assert cuts == [20]


def test_single_image_path_needs_no_stream():
    fs = Gate().image(synth.hsv_scene(60, 2))
    for k in [
        "exposure",
        "contrast",
        "colorfulness",
        "detail",
        "clipping",
        "noise_floor",
    ]:
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
            _, sig = g.frame(f)
            res.append(
                (
                    sig.cut,
                    round(sig.cut_score, 6),
                    sig.freeze,
                    round(sig.struct_corr, 6),
                )
            )
        return res

    assert outputs(True) == outputs(False)


def test_return_frames_default_on_and_toggleable():
    import cv2

    t = GateConfig().thumb
    fs = Gate().image(synth.hsv_scene(60, 2))
    assert fs.thumb.shape == (t, t, 3) and fs.hsv.shape == (t, t, 3)
    # hsv is made from the thumbnail on demand: cvtColor's bytes
    assert np.array_equal(fs.hsv, cv2.cvtColor(fs.thumb, cv2.COLOR_BGR2HSV))
    g = Gate().image(synth.grayscale_scene(2))
    assert g.thumb.shape == (t, t) and g.hsv.shape == (
        t,
        t,
        3,
    )  # grayscale thumb is 1-channel
    assert not g.hsv[:, :, :2].any() and np.array_equal(g.hsv[:, :, 2], g.thumb)
    off = Gate(GateConfig(return_frames=False)).image(synth.hsv_scene(60, 2))
    assert off.thumb is None and off.hsv is None


def test_pass_on_bgr_equals_pass_on_cvtcolor_hsv():
    """imfeat converts the thumbnail to HSV inside the gate's pass; every number the gate
    reads is byte for byte what a pass on the cvtColor'd thumbnail gives, so signals and
    models behave exactly as before the conversion moved. A grayscale frame likewise
    (H = S = 0, V = luma)."""
    import imfeat  # type: ignore[import-untyped]

    from framegate import signals as S

    cfg = GateConfig()
    gate = Gate(cfg)
    as_is = imfeat.FeatureComputer(
        shape=(cfg.thumb, cfg.thumb, 3),
        grid=[(e, e) for e in cfg.pyramid_exps],
        stride=cfg.stride,
        feature_space=None,
    )
    for frame in (synth.noisy(synth.hsv_scene(60, 2)), synth.grayscale_scene(2)):
        fs = gate.image(frame)
        want = as_is.features(fs.hsv)  # the old path: cvtColor first, imfeat on HSV
        assert np.array_equal(fs.chan, want.moments[-1].astype(np.float32))
        for grid, level in zip(fs.grids, want.moments[: cfg.n_levels], strict=True):
            assert np.array_equal(grid, level.astype(np.float32))
        assert fs.phash == int(want.hashes[imfeat.HASHES.index("phash"), S.CH_V])
        fine = want.maps[0].reshape(
            *want.maps[0].shape[:2], 3, len(imfeat.FEATURE_NAMES)
        )
        assert np.array_equal(fs.struct["grid_0"], fine[:, :, S.CH_V, S.SE])


def test_pass_on_frame_equals_pass_on_cv2_thumbnail():
    """imfeat thumbnails a frame inside the gate's pass (cv2.resize(INTER_AREA)'s bytes);
    every number the gate reads, and the thumbnail it hands back, are byte for byte what
    cv2.resize followed by the pass on the thumbnail gives. Frames cv2 has to resize --
    smaller than the thumbnail in an axis (bilinear in OpenCV), grayscale, "nearest" --
    still come out that way, and a change of frame size mid-stream is fine."""
    import cv2
    import imfeat

    from framegate import signals as S

    def check(gate, frame, interp):
        cfg = gate.cfg
        as_is = imfeat.FeatureComputer(
            shape=(cfg.thumb, cfg.thumb, 3),
            grid=[(e, e) for e in cfg.pyramid_exps],
            stride=cfg.stride,
            feature_space=None,
        )
        fs = gate.image(frame)
        small = cv2.resize(frame, (cfg.thumb, cfg.thumb), interpolation=interp)
        if small.ndim == 2:  # a grayscale frame: H = S = 0, V = luma
            assert np.array_equal(fs.thumb, small)
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        else:
            assert np.array_equal(fs.thumb, small)
        want = as_is.features(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
        assert np.array_equal(fs.chan, want.moments[-1].astype(np.float32))
        for grid, level in zip(fs.grids, want.moments[: cfg.n_levels], strict=True):
            assert np.array_equal(grid, level.astype(np.float32))
        assert fs.phash == int(want.hashes[imfeat.HASHES.index("phash"), S.CH_V])
        fine = want.maps[0].reshape(
            *want.maps[0].shape[:2], 3, len(imfeat.FEATURE_NAMES)
        )
        assert np.array_equal(fs.struct["grid_0"], fine[:, :, S.CH_V, S.SE])
        assert np.array_equal(
            fs.struct["global"], want.maps[-1].reshape(3, -1)[S.CH_V, S.SE]
        )

    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:1080, 0:1920]
    frames = [
        rng.integers(
            0, 256, (1080, 1920, 3), dtype=np.uint8
        ),  # fused: the frame goes in
        np.stack(
            [xx * 255 // 1919, yy * 255 // 1079, ((xx // 7 + yy // 5) % 2) * 255], -1
        ).astype(np.uint8),
        rng.integers(0, 256, (1440, 2560, 3), dtype=np.uint8),  # another size: rebuilt
        rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8),  # cv2: an upscale
        rng.integers(0, 256, (1080, 1920), dtype=np.uint8),  # cv2: grayscale
        rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8),  # fused again
    ]
    gate = Gate(GateConfig(feat_threads=2))
    for frame in frames:
        check(gate, frame, cv2.INTER_AREA)
    gate = Gate(GateConfig(resize_interp="nearest"))  # cv2 throughout
    for frame in frames[:1] + frames[3:4]:
        check(gate, frame, cv2.INTER_NEAREST)


def test_fast_static_matches_full_search_on_cuts():
    a = [synth.noisy(synth.hsv_scene(60, 2)) for _ in range(20)]
    b = [synth.noisy(synth.hsv_scene(60, 3)) for _ in range(8)]
    on = _cuts(a + b, GateConfig(fast_static=True))
    off = _cuts(a + b, GateConfig(fast_static=False))
    assert on == off == [20]


def test_appearance_maps_cached_and_reused_on_duplicate():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    g = Gate()
    fs1, _ = g.frame(img)
    assert fs1.saliency is fs1.saliency  # appearance maps cached (pure per-frame)
    assert fs1.text is fs1.text
    fs2, _ = g.frame(img.copy())  # byte-identical -> stats reused
    assert fs2 is fs1  # duplicate-skip returns the same object
    assert fs2.saliency is fs1.saliency  # reused for free


def test_motion_map_only_on_video_and_denoised():
    G = GateConfig().grid_size

    def frame_at(x):  # a bright block that moves each frame
        f = np.full((240, 420, 3), 30, np.uint8)
        f[60:180, x : x + 120] = 220
        return f

    assert Gate().image(frame_at(100)).motion is None  # a still image has no motion

    g = Gate()
    for x in (40, 70, 100, 130, 160):
        fs, _ = g.frame(frame_at(x))
    assert fs.motion.shape == (G, G)  # (G,G) magnitude map
    assert fs.motion.min() >= 0.0 and fs.motion.max() > 0.0  # movement registers

    # the noise floor suppresses static sensor speckle: a near-static noisy stream -> mostly zeros
    rng = np.random.default_rng(5)
    base = rng.integers(110, 140, (240, 420, 3), dtype=np.uint8)
    gq = Gate()
    for _ in range(4):
        noisy = np.clip(
            base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255
        ).astype(np.uint8)
        fsq, _ = gq.frame(noisy)
    cleaned = float((fsq.motion == 0).mean())
    raw = float((np.abs(fsq.residual) == 0).mean())
    assert cleaned > raw  # denoise zeroes speckle the raw map keeps
    assert cleaned > 0.5  # most of a static noisy frame reads as no-motion
