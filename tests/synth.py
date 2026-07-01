"""Synthetic frame builders shared by the tests. Kept deterministic so accuracy
assertions are stable."""

import cv2
import numpy as np

THUMB = 128
_rng = np.random.default_rng(0)


def hsv_scene(hue, vseed, sat=200, size=THUMB):
    """A structured-luma scene with a (flat) hue and high saturation."""
    r = np.random.default_rng(vseed)
    V = np.clip(r.random((size, size)) * 120 + 80, 0, 255).astype(np.uint8)
    H = np.full((size, size), hue, np.uint8)
    Sx = np.full((size, size), sat, np.uint8)
    return cv2.cvtColor(cv2.merge([H, Sx, V]), cv2.COLOR_HSV2BGR)


def noisy(img, amp=3):
    return np.clip(
        img.astype(np.int16) + _rng.integers(-amp, amp + 1, img.shape), 0, 255
    ).astype(np.uint8)


def dim(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= factor
    return cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def black(size=THUMB):
    return np.zeros((size, size, 3), np.uint8)


def white(size=THUMB):
    return np.full((size, size, 3), 255, np.uint8)


def solid(value, size=THUMB):
    """A uniform BGR frame (value may be an int or a (B,G,R) triple)."""
    return np.full((size, size, 3), value, np.uint8)


def noise(size=THUMB, seed=0):
    """Full-range random colour noise (high detail, not blank)."""
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def gradient(size=THUMB, axis=1):
    """A smooth luminance ramp (coarse structure, little fine texture)."""
    ramp = np.linspace(0, 255, size, dtype=np.float32)
    g = np.tile(ramp, (size, 1)) if axis == 1 else np.tile(ramp[:, None], (1, size))
    return cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def checkerboard(size=THUMB, cell=8):
    """A black/white checkerboard (strong fine texture, achromatic)."""
    yy, xx = np.mgrid[0:size, 0:size]
    g = (((yy // cell) + (xx // cell)) % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def text_block(size=256, scale=0.5):
    """Lines of real rendered text (mixed-orientation strokes, dark ink on light
    paper) -- a realistic text proxy, unlike single-orientation bars. Rendered at
    thumb resolution so strokes survive resizing."""
    img = np.full((size, size), 235, np.uint8)
    for y in range(24, size - 6, 26):
        cv2.putText(
            img,
            "The quick brown fox",
            (6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            20,
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def rules(size=256):
    """Horizontal rule lines on paper (underlines / table borders): a coherent,
    bimodal, achromatic oriented pattern -- a hard text false-positive."""
    img = np.full((size, size), 235, np.uint8)
    for y in range(20, size - 6, 30):
        img[y : y + 2, 10 : size - 10] = 20
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def stripes(size=THUMB, period=4, horizontal=True):
    """Fine achromatic line pattern (text-like high-frequency horizontal texture)."""
    idx = (np.arange(size) // (period // 2)) % 2 * 255
    g = np.tile(idx, (size, 1)) if horizontal else np.tile(idx[:, None], (1, size))
    return cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def moving_block(x, size=THUMB, w=None, bg=30, fg=220):
    """A bright block on a dark field, left edge at column `x` (for motion tests)."""
    w = w or size // 3
    f = np.full((size, size, 3), bg, np.uint8)
    f[size // 3 : 2 * size // 3, x : x + w] = fg
    return f


def fade_ramp(scene, factors):
    """A list of frames dimming/brightening `scene` by each factor in turn."""
    return [noisy(dim(scene, f)) for f in factors]


def grayscale_scene(vseed, size=THUMB):
    r = np.random.default_rng(vseed)
    return np.clip(r.random((size, size)) * 120 + 80, 0, 255).astype(np.uint8)


def run_stream(gate_or_stream, frames, gate=None):
    """Drive a sequence and collect (cuts, freezes, fades, flickers).
    If `gate` is given, frames are FrameStats producers; otherwise a Gate is used."""
    from framegate import Gate

    g = gate_or_stream if isinstance(gate_or_stream, Gate) else Gate()
    cuts, freezes, fades, flickers = [], 0, [], []
    for f in frames:
        _, sig = g.frame(f)
        if sig.cut:
            cuts.append(sig.cut_frame)
        if sig.freeze:
            freezes += 1
        fades.append(sig.fade)
        flickers.append(sig.flicker)
    return cuts, freezes, fades, flickers
