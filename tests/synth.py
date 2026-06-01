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
    return np.clip(img.astype(np.int16) + _rng.integers(-amp, amp + 1, img.shape), 0, 255).astype(np.uint8)


def dim(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= factor
    return cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def black(size=THUMB):
    return np.zeros((size, size, 3), np.uint8)


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
