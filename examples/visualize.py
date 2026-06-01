"""Live visualization of framegate on a video. Pure consumer of the public API
(nothing here is imported by the library itself).

    python examples/visualize.py path/to/video.mp4

Requires the [viz] extra:  pip install "framegate[viz]"
"""

import sys
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from framegate import Gate

HISTORY = 300        # rolling plot window (viz only)
DISPLAY_MAX = 512    # longest side of the displayed frame
GRID_CMAPS = ["inferno", "viridis", "coolwarm", "plasma"]
MOMENT_LBL = ["mean", "variance", "m3", "m4"]
TS_PLOTS = [("exposure", "V mean"), ("contrast", "V std"), ("colorfulness", "S mean"),
            ("detail", "detail"), ("struct_corr", "struct corr"), ("cut_score", "cut score"),
            ("fade", "fade"), ("flicker", "flicker")]


def _grid_row(axes, label, g):
    ims = []
    for ax, lbl, cmap in zip(axes, MOMENT_LBL, GRID_CMAPS):
        im = ax.imshow(np.zeros((g, g)), cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_title(f"{label} {lbl}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ims.append(im)
    return ims


def _imshow(ax, g, cmap, title):
    ax.set_title(title, fontsize=8)
    return ax.imshow(np.zeros((g, g)), cmap=cmap, aspect="auto")


def run(src):
    cap = cv2.VideoCapture(src)
    assert cap.isOpened(), f"cannot open {src}"
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, DISPLAY_MAX / max(sw, sh))
    W, H = max(1, int(sw * scale)), max(1, int(sh * scale))

    gate = Gate()
    g = gate.cfg.grid_size
    hist = {k: deque([0.0] * HISTORY, maxlen=HISTORY) for k, _ in TS_PLOTS}
    cut_frames = []

    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(7, 4, figure=fig, hspace=0.55, wspace=0.35,
                           left=0.05, right=0.97, top=0.94, bottom=0.04)
    ax_frame = fig.add_subplot(gs[0, :2]); ax_frame.axis("off")
    ax_status = fig.add_subplot(gs[0, 2:]); ax_status.axis("off")
    gV = _grid_row([fig.add_subplot(gs[1, c]) for c in range(4)], "V", g)
    gH = _grid_row([fig.add_subplot(gs[2, c]) for c in range(4)], "H", g)
    gS = _grid_row([fig.add_subplot(gs[3, c]) for c in range(4)], "S", g)
    ax_m = fig.add_subplot(gs[4, 0]); ax_s = fig.add_subplot(gs[4, 1])
    ax_t = fig.add_subplot(gs[4, 2]); ax_tl = fig.add_subplot(gs[4, 3])
    im_m = _imshow(ax_m, g, "hot", "motion (illum-invariant)")
    im_s = _imshow(ax_s, g, "magma", "saliency")
    im_t = _imshow(ax_t, g, "cividis", "fine texture")
    ax_tl.set_title("cut score (cut = red line)", fontsize=8); ax_tl.set_xlim(0, HISTORY)
    (ln_cs,) = ax_tl.plot(np.zeros(HISTORY), color="C3", lw=1.0)
    ax_ts = [fig.add_subplot(gs[5 + i // 4, i % 4]) for i in range(len(TS_PLOTS))]

    ret, first = cap.read()
    disp0 = cv2.cvtColor(cv2.resize(first, (W, H)), cv2.COLOR_BGR2RGB)
    im_frame = ax_frame.imshow(disp0, aspect="auto")
    banner = ax_frame.text(0.5, 0.92, "", ha="center", va="center", fontsize=22, fontweight="bold",
                           color="white", transform=ax_frame.transAxes,
                           bbox=dict(boxstyle="round", fc="black", alpha=0.6))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    lines = {}
    for ax, (k, lbl) in zip(ax_ts, TS_PLOTS):
        (ln,) = ax.plot(np.arange(HISTORY), np.zeros(HISTORY), lw=1.0)
        ax.set_title(lbl, fontsize=8); ax.set_xlim(0, HISTORY); ax.tick_params(labelsize=7)
        lines[k] = (ax, ln)
    ax_ts[5].axhline(gate.cfg.cut_dissim, color="r", ls=":", lw=0.8)

    plt.ion(); plt.show()
    fidx = 0
    while plt.fignum_exists(fig.number):
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1
        fs, sig = gate.frame(frame)
        for k in hist:
            hist[k].append(getattr(fs, k) if hasattr(fs, k) else getattr(sig, k))
        if sig.cut:
            cut_frames.append(fidx - 1)

        im_frame.set_data(cv2.cvtColor(cv2.resize(frame, (W, H)), cv2.COLOR_BGR2RGB))
        banner.set_text("BLANK" if fs.blank else "CUT" if sig.cut else "FREEZE" if sig.freeze else "")

        ax_status.clear(); ax_status.axis("off")
        ax_status.text(0.5, .82, "BLANK" if fs.blank else "active", ha="center",
                       color="tomato" if fs.blank else "limegreen", fontsize=16, fontweight="bold")
        ax_status.text(0.5, .58, f"corr={sig.struct_corr:.3f}   gain={sig.gain:.2f}   bias={sig.bias:+.1f}",
                       ha="center", fontsize=10, color="gray")
        ax_status.text(0.5, .43, f"cut_score={sig.cut_score:.3f}   fade={sig.fade:+.2f}   flicker={sig.flicker:.2f}",
                       ha="center", fontsize=10, color="gray")
        ax_status.text(0.5, .28, f"noise={fs.noise_floor:.2f}   clip={fs.clipping:+.2f}   colorful={fs.colorfulness:.1f}",
                       ha="center", fontsize=10, color="gray")

        for ims, ch in ((gV, "grid_V"), (gH, "grid_H"), (gS, "grid_S")):
            chan = getattr(fs, ch)
            for i, im in enumerate(ims):
                d = chan[:, :, i]; im.set_data(d); im.set_clim(d.min(), d.max() + 1e-6)
        g_z = np.zeros((g, g), np.float32)
        for im, d in ((im_m, fs.motion if fs.motion is not None else g_z),
                      (im_s, fs.saliency), (im_t, fs.fine_texture)):
            im.set_data(d); im.set_clim(0, d.max() + 1e-6)

        cs = np.asarray(hist["cut_score"]); ln_cs.set_ydata(cs)
        ax_tl.set_ylim(0, max(cs.max(), gate.cfg.cut_dissim) * 1.1 + 1e-3)
        [c.remove() for c in list(ax_tl.lines[1:])]
        origin = fidx - HISTORY + 1
        for f in cut_frames:
            if origin <= f <= fidx:
                ax_tl.axvline(f - origin, color="red", lw=1.0, alpha=.7)
        for k, (ax, ln) in lines.items():
            d = np.asarray(hist[k]); ln.set_ydata(d)
            lo, hi = d.min(), d.max(); ax.set_ylim(lo - max((hi - lo) * .1, .05), hi + max((hi - lo) * .1, .05))

        fig.suptitle(f"frame {fidx}  |  {src}", fontsize=10)
        fig.canvas.draw_idle(); fig.canvas.flush_events()

    cap.release(); plt.ioff(); plt.show()
    print(f"done. {fidx} frames.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/visualize.py <video.mp4>"); sys.exit(1)
    run(sys.argv[1])
