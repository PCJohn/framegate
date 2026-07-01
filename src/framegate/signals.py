"""Numeric core: small, pure, stateless functions over the moment grids.

Everything here is side-effect free and depends only on numpy/cv2, so each
function is independently testable and trivial to port. Channel/moment layout
constants live here because this is the lowest layer that indexes the grids.
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# tensorstats layout: result["0,1"] is (3, 4) per-channel moments; result["grid_0"]
# is (G, G, C, 4) = [row, col, channel, moment].
CH_H, CH_S, CH_V = 0, 1, 2
M_MEAN, M_VAR, M_M3, M_M4 = 0, 1, 2, 3

# structstats.features() channel layout (ss.FEATURES order).
SE_ENERGY, SE_COH, SE_OC, SE_OS, SE_CORN = 0, 1, 2, 3, 4


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    if s < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x.mean()) / s).astype(np.float32)


def box(x: np.ndarray, kx: int, ky: int) -> np.ndarray:
    """Normalized box filter (kx wide, ky tall), replicating the border."""
    return cv2.boxFilter(
        x, -1, (kx, ky), normalize=True, borderType=cv2.BORDER_REPLICATE
    )


def ncc0(prev: np.ndarray, cur: np.ndarray) -> float:
    """Zero-shift normalized cross-correlation of two equal-shape maps."""
    a, b = prev - prev.mean(), cur - cur.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def best_shift(prev: np.ndarray, cur: np.ndarray, s: int):
    """argmax normalized cross-correlation of cur's center vs prev over +/-s
    integer cell shifts (square maps), so a camera pan (a translation of the map)
    still correlates highly and isn't read as a cut. Returns (corr, dy, dx),
    vectorized over all (2s+1)^2 shifts via a sliding-window view."""
    if s <= 0:
        return ncc0(prev, cur), 0, 0
    g = prev.shape[-1]
    h = g - 2 * s
    c = cur[s : s + h, s : s + h]
    c = c - c.mean()
    nc = np.sqrt((c * c).sum()) + 1e-6
    wins = sliding_window_view(prev, (h, h)).reshape(-1, h, h)
    a = wins - wins.mean(axis=(1, 2), keepdims=True)
    na = np.sqrt((a * a).sum(axis=(1, 2))) + 1e-6
    corr = (a * c).sum(axis=(1, 2)) / (na * nc)
    k = int(corr.argmax())
    n = 2 * s + 1
    return float(corr[k]), k // n - s, k % n - s


def saliency_map(grid_V, grid_S, surround_k: int) -> np.ndarray:
    """Coarse (G,G) appearance saliency from per-cell stats: V-variance (texture) +
    S-mean (colorfulness) + center-surround luma contrast, z-scored and averaged, clipped
    at 0. The luma term is |V-mean - local mean| over a `surround_k` neighborhood (a box
    blur), so a cell is judged against its surround rather than the global frame mean --
    the standard bottom-up center-surround principle. Purely per-frame; motion is separate.
    """
    vmean = grid_V[:, :, M_MEAN].astype(np.float32)
    v_con = np.abs(vmean - box(vmean, surround_k, surround_k))
    feats = [zscore(grid_V[:, :, M_VAR]), zscore(grid_S[:, :, M_MEAN]), zscore(v_con)]
    return np.mean(feats, axis=0).clip(min=0).astype(np.float32)


def text(
    grid_V,
    grid_S,
    coherence,
    achromatic_w: float,
    coarse_k: int,
    line_k: int,
    skew_w: float,
    skew_ref: float,
    coh_w: float,
) -> np.ndarray:
    """Per-cell text likelihood from low-level texture. High-frequency within-cell
    contrast NOT explained by coarse between-cell variation (a multi-scale high-pass),
    down-weighted by saturation, gated by per-cell distribution asymmetry, gated by
    gradient isotropy, and smoothed horizontally. Two complementary, orthogonal gates
    make this text-specific: (1) the asymmetry gate -- text is bimodal (sparse ink on
    paper), so its per-cell |standardized skew| is high while isotropic clutter (foliage,
    noise) is symmetric and is suppressed; (2) the isotropy gate -- text is a mix of
    stroke orientations within a cell (low structure-tensor coherence), so coherent
    oriented patterns (single edges, rules, fences, hatching) that survive the high-pass
    are suppressed. Tuned for dense, achromatic, horizontally-laid-out text (printed body
    text, captions, dense UI); large display or colourful text score lower. A cue, not OCR.
    """
    var = np.maximum(grid_V[:, :, M_VAR], 0.0)
    fine = np.sqrt(var).astype(np.float32)
    mean = grid_V[:, :, M_MEAN].astype(np.float32)
    coarse = np.sqrt(
        np.maximum(
            box(mean * mean, coarse_k, coarse_k) - box(mean, coarse_k, coarse_k) ** 2,
            0.0,
        )
    )
    # Bimodality gate: |standardized skew| = |m3| / var**1.5 (var**1.5 = var * fine).
    # ~1.6 for text, ~0.2-0.7 for symmetric clutter; absolute scale, not per-frame.
    skew = np.abs(grid_V[:, :, M_M3]) / (var * fine + 1e-6)
    bimodal = 1.0 - skew_w * (1.0 - np.minimum(skew / skew_ref, 1.0))
    # Isotropy gate: coherence ~0.5 for mixed-orientation text, ~1 for a single
    # dominant edge / parallel rules; down-weight the coherent (non-text) cells.
    iso = np.clip(1.0 - coh_w * coherence, 0.0, 1.0)
    score = (
        np.maximum(fine - coarse, 0.0)
        * (1.0 - achromatic_w * grid_S[:, :, M_MEAN] / 255.0)
        * bimodal
        * iso
    )
    return box(score, line_k, 1)


def fade_score(series: np.ndarray, span: float) -> float:
    """Signed fade strength in [-1, 1] over a brightness series: sign = direction
    (negative = darkening), magnitude = monotonicity * normalized span."""
    d = np.diff(series)
    total = float(series[-1] - series[0])
    if total == 0.0:
        return 0.0
    mono = float(np.mean(np.sign(d) == np.sign(total)))
    return float(np.sign(total) * mono * min(abs(total) / span, 1.0))


def flicker_score(series: np.ndarray, window: np.ndarray) -> float:
    """Dominant-frequency power fraction of the detrended brightness series in
    [0, 1]: high = a strong periodic component (strobe / flicker). `window` is a
    precomputed taper (e.g. Hann) the length of `series`."""
    x = series - series.mean()
    if x.std() < 1e-3:
        return 0.0
    p = np.abs(np.fft.rfft(x * window)) ** 2
    p[0] = 0.0
    return float(p.max() / (p.sum() + 1e-9))
