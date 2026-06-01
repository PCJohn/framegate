"""Numeric core: small, pure, stateless functions over the moment grids.

Everything here is side-effect free and depends only on numpy/cv2, so each
function is independently testable and trivial to port. Channel/moment layout
constants live here because this is the lowest layer that indexes the grids.
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# tensorstats layout: result["0,1"] is (3, 4) per-channel moments; result["grid"]
# is (G, G, C, 4) = [row, col, channel, moment].
CH_H, CH_S, CH_V = 0, 1, 2
M_MEAN, M_VAR, M_M3, M_M4 = 0, 1, 2, 3


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    if s < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x.mean()) / s).astype(np.float32)


def box(x: np.ndarray, kx: int, ky: int) -> np.ndarray:
    """Normalized box filter (kx wide, ky tall), replicating the border."""
    return cv2.boxFilter(x, -1, (kx, ky), normalize=True, borderType=cv2.BORDER_REPLICATE)


def best_shift(prev: np.ndarray, cur: np.ndarray, s: int):
    """argmax normalized cross-correlation of cur's center vs prev over +/-s
    integer cell shifts (square maps), so a camera pan (a translation of the map)
    still correlates highly and isn't read as a cut. Returns (corr, dy, dx),
    vectorized over all (2s+1)^2 shifts via a sliding-window view."""
    if s <= 0:
        a, b = prev - prev.mean(), cur - cur.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6)), 0, 0
    g = prev.shape[-1]; h = g - 2 * s
    c = cur[s:s + h, s:s + h]; c = c - c.mean(); nc = np.sqrt((c * c).sum()) + 1e-6
    wins = sliding_window_view(prev, (h, h)).reshape(-1, h, h)
    a = wins - wins.mean(axis=(1, 2), keepdims=True)
    na = np.sqrt((a * a).sum(axis=(1, 2))) + 1e-6
    corr = (a * c).sum(axis=(1, 2)) / (na * nc)
    k = int(corr.argmax()); n = 2 * s + 1
    return float(corr[k]), k // n - s, k % n - s


def saliency_map(grid_V, grid_S) -> np.ndarray:
    """Coarse (G,G) appearance saliency from per-cell stats: V-variance (texture) +
    S-mean (colorfulness) + |V-mean - frame mean| (luma contrast), z-scored and
    averaged, clipped at 0. Purely per-frame -- motion is a separate signal, not
    folded in here."""
    v_con = np.abs(grid_V[:, :, M_MEAN] - grid_V[:, :, M_MEAN].mean())
    feats = [zscore(grid_V[:, :, M_VAR]), zscore(grid_S[:, :, M_MEAN]), zscore(v_con)]
    return np.mean(feats, axis=0).clip(min=0).astype(np.float32)


def fine_texture(grid_V, grid_S, achromatic_w: float, coarse_k: int, line_k: int) -> np.ndarray:
    """Per-cell fine-texture energy: high-frequency within-cell contrast that is NOT
    explained by coarse between-cell variation (a multi-scale high-pass), down-weighted
    by saturation and smoothed horizontally. A generic, script-agnostic cue for text /
    print / dense UI -- patterns that are fine-scale, achromatic, and in horizontal runs
    -- but a low-level texture descriptor, not OCR. Distinct from raw V-variance, which
    also counts coarse structure."""
    fine = np.sqrt(np.maximum(grid_V[:, :, M_VAR], 0.0)).astype(np.float32)
    mean = grid_V[:, :, M_MEAN].astype(np.float32)
    coarse = np.sqrt(np.maximum(box(mean * mean, coarse_k, coarse_k) - box(mean, coarse_k, coarse_k) ** 2, 0.0))
    score = np.maximum(fine - coarse, 0.0) * (1.0 - achromatic_w * grid_S[:, :, M_MEAN] / 255.0)
    return box(score, line_k, 1)


def _bbox_stack(mask: np.ndarray, shape) -> list:
    """For each (G,G) mask in a (K,G,G) stack, the pixel-space bounding box
    (x0,y0,x1,y1) of its True cells, or None if empty."""
    k, g, _ = mask.shape
    sy, sx = shape[0] / g, shape[1] / g
    rows, cols = mask.any(axis=2), mask.any(axis=1)
    out = []
    for i in range(k):
        r, c = np.where(rows[i])[0], np.where(cols[i])[0]
        out.append(None if r.size == 0 else
                   (int(c[0] * sx), int(r[0] * sy), int((c[-1] + 1) * sx), int((r[-1] + 1) * sy)))
    return out


def roi_boxes(stack: np.ndarray, shape, k: float):
    """Trim-based bounding boxes for every map in a (K,G,G) stack, in both
    polarities: cells above mean + k*std ('hi', the bright/active region) and
    below mean - k*std ('lo', the dark/inverse region). Each map is judged by its
    own statistics, so one cheap pass yields a diverse set of candidate ROIs that
    catch subjects a single map would miss. Returns (hi_boxes, lo_boxes)."""
    mu = stack.mean(axis=(1, 2), keepdims=True)
    sd = stack.std(axis=(1, 2), keepdims=True) + 1e-6
    return _bbox_stack(stack > mu + k * sd, shape), _bbox_stack(stack < mu - k * sd, shape)


def merge_boxes(boxes, labels, iou_thr: float):
    """Reconcile overlapping candidate boxes into one labelled list. All-pairs IoU is
    computed once (vectorized), then a greedy pass keeps the box with the most overlap
    support as the representative and absorbs everything it overlaps, accumulating their
    labels. Returns [(box, [labels...]), ...]; the label list records which maps proposed
    the region. Cheap because the set is tiny."""
    if not boxes:
        return []
    b = np.asarray(boxes, np.float32)
    x1, y1, x2, y2 = b.T
    area = (x2 - x1) * (y2 - y1)
    xx1 = np.maximum(x1[:, None], x1); yy1 = np.maximum(y1[:, None], y1)
    xx2 = np.minimum(x2[:, None], x2); yy2 = np.minimum(y2[:, None], y2)
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    iou = inter / (area[:, None] + area - inter + 1e-9)
    adj = iou > iou_thr
    used = np.zeros(len(boxes), bool)
    out = []
    for i in adj.sum(1).argsort()[::-1]:        # most-overlapping seed first
        if used[i]:
            continue
        members = adj[i] & ~used
        used |= members
        out.append((boxes[i], sorted(labels[j] for j in np.where(members)[0])))
    return out


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
