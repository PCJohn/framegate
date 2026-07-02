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


FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
)


def text_block(size=256, scale=0.5, font=0, fg=20, bg=235, thick=1, rot=0.0):
    """Lines of real rendered text (mixed-orientation strokes, dark ink on light
    paper) -- a realistic text proxy, unlike single-orientation bars. Parametric over
    font, colour, weight and rotation so tests can sweep realistic variation. `fg`/`bg`
    may be a scalar grey or a BGR triple. Rendered at thumb resolution so strokes
    survive resizing."""
    fg = fg if isinstance(fg, (tuple, list)) else (fg, fg, fg)
    bg = bg if isinstance(bg, (tuple, list)) else (bg, bg, bg)
    img = np.full((size, size, 3), bg, np.uint8)
    for y in range(24, size - 6, 26):
        cv2.putText(
            img,
            "The quick brown fox",
            (6, y),
            FONTS[font],
            scale,
            fg,
            thick,
            cv2.LINE_AA,
        )
    if rot:
        m = cv2.getRotationMatrix2D((size / 2, size / 2), rot, 1.0)
        img = cv2.warpAffine(img, m, (size, size), borderValue=bg)
    return img


def rules(size=256):
    """Horizontal rule lines on paper (underlines / table borders): a coherent,
    bimodal, achromatic oriented pattern -- a hard text false-positive."""
    img = np.full((size, size), 235, np.uint8)
    for y in range(20, size - 6, 30):
        img[y : y + 2, 10 : size - 10] = 20
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def foliage(size=256, seed=1):
    """Organic multi-scale achromatic clutter (foliage/gravel-like): symmetric,
    isotropic, high-frequency -- the canonical non-text texture distractor."""
    rng = np.random.default_rng(seed)
    coarse = np.repeat(
        np.repeat(rng.integers(40, 210, (20, 20)), size // 20 + 1, 0), size // 20 + 1, 1
    )
    g = (0.6 * coarse[:size, :size] + 0.4 * rng.integers(0, 255, (size, size))).astype(
        np.uint8
    )
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def orientation_popout(size=256, band=3, patch=(96, 160)):
    """A field of horizontal stripes with a square patch of vertical stripes, identical
    in contrast, mean and colour -- a classic orientation pop-out: salient to humans and
    to an orientation-contrast channel, invisible to intensity/colour/variance channels.
    """
    col = ((np.arange(size) // band) % 2 * 180 + 40).astype(np.uint8)
    horiz = np.tile(col[:, None], (1, size))
    vert = np.tile(col[None, :], (size, 1))
    img = horiz.copy()
    a, b = patch
    img[a:b, a:b] = vert[a:b, a:b]
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
