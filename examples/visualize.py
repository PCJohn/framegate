"""Live dashboard for framegate on a video. Pure consumer of the public API
(nothing here is imported by the library itself).

    python examples/visualize.py path/to/video.mp4

Shows the appearance maps (motion / saliency / text / focus / structure), the per-cell
moment grids, the temporal event signals (cut / fade / flicker / struct-corr),
and a live latency panel separating framegate compute from matplotlib render --
so the speed of the package is visible against the cost of just drawing it.

Requires the [viz] extra:  pip install "framegate[viz]"
"""

import gc
import sys
import time
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

from framegate import Gate, ShotTracker

HISTORY = 300  # rolling time-series window (viz only)
DRAW_EVERY = 3  # redraw the dashboard every N frames; compute still runs every frame
DISPLAY_MAX = 480  # longest side of the displayed frame
LAT_WIN = 30  # frames to average for the latency readout
# Fixed display ranges for the map panels, so a near-static frame stays dark
# instead of auto-stretching its noise floor to full brightness.
MAP_VMAX = {"motion": 32.0, "saliency": 3.0, "texture": 24.0, "focus": 30.0}


def heat(ax, title, cmap, clim=None):
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(
        np.zeros((2, 2), np.float32), cmap=cmap, aspect="auto", interpolation="nearest"
    )
    if clim:
        im.set_clim(*clim)
    return im


def tseries(ax, title, labels, colors):
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, HISTORY)
    ax.tick_params(labelsize=6)
    lns = [
        ax.plot(np.zeros(HISTORY), lw=1.0, color=c, label=lab)[0]
        for lab, c in zip(labels, colors)
    ]
    if len(labels) > 1:
        ax.legend(fontsize=6, loc="upper left", ncol=len(labels), framealpha=0.4)
    return lns


def run(src):
    cap = cv2.VideoCapture(src)
    assert cap.isOpened(), f"cannot open {src}"
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    scale = min(1.0, DISPLAY_MAX / max(sw, sh))
    W, H = max(1, int(sw * scale)), max(1, int(sh * scale))

    gate = Gate()
    g = gate.cfg.grid_size
    tracker = ShotTracker(gate.cfg)  # shot_id + shot_group_id

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(
        4, 7, hspace=0.5, wspace=0.3, left=0.03, right=0.99, top=0.93, bottom=0.05
    )

    ax_frame = fig.add_subplot(gs[0:2, 0:3])
    ax_frame.axis("off")
    im_frame = ax_frame.imshow(np.zeros((H, W, 3), np.uint8), aspect="auto")
    banner = ax_frame.text(
        0.5,
        0.93,
        "",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="white",
        transform=ax_frame.transAxes,
        bbox=dict(boxstyle="round", fc="black", alpha=0.6),
    )

    # appearance maps (fixed clim, no colorbars)
    im_mot = heat(
        fig.add_subplot(gs[0, 3]), "motion (illum-inv)", "hot", (0, MAP_VMAX["motion"])
    )
    im_sal = heat(
        fig.add_subplot(gs[0, 4]), "saliency", "magma", (0, MAP_VMAX["saliency"])
    )
    im_tex = heat(
        fig.add_subplot(gs[0, 5]), "text", "cividis", (0, MAP_VMAX["texture"])
    )
    # exact per-cell moment grids (autoscaled)
    im_luma = heat(fig.add_subplot(gs[1, 3]), "luma  (V mean)", "inferno")
    im_var = heat(fig.add_subplot(gs[1, 4]), "contrast  (V var)", "viridis")
    im_sat = heat(fig.add_subplot(gs[1, 5]), "saturation  (S mean)", "plasma")
    im_foc = heat(
        fig.add_subplot(gs[0, 6]),
        "focus (edge sharpness)",
        "bone",
        (0, MAP_VMAX["focus"]),
    )
    ax_st = fig.add_subplot(gs[1, 6])
    ax_st.set_title("structure  (flat/edge/tex = RGB)", fontsize=8)
    ax_st.set_xticks([])
    ax_st.set_yticks([])
    im_st = ax_st.imshow(
        np.zeros((2, 2, 3), np.float32), aspect="auto", interpolation="nearest"
    )

    # cut-score timeline with threshold + cut markers
    ax_cut = fig.add_subplot(gs[2, 0:3])
    ax_cut.set_title("cut score   (threshold dotted, cut = red)", fontsize=8)
    ax_cut.set_xlim(0, HISTORY)
    ax_cut.tick_params(labelsize=6)
    (ln_cut,) = ax_cut.plot(np.zeros(HISTORY), color="C3", lw=1.1)
    ax_cut.axhline(gate.cfg.cut_dissim, color="r", ls=":", lw=0.8)

    # latency timeline: framegate compute vs matplotlib render
    ax_lat = fig.add_subplot(gs[2, 3:7])
    ln_lat = tseries(
        ax_lat, "framegate latency (ms/frame)", ["core", "core+maps"], ["C0", "C1"]
    )

    # temporal event signals
    ax_ev = fig.add_subplot(gs[3, 0:2])
    ax_ev.set_ylim(-1.05, 1.05)
    ln_ev = tseries(
        ax_ev, "events", ["struct_corr", "fade", "flicker"], ["C2", "C0", "C4"]
    )

    ax_txt = fig.add_subplot(gs[3, 2:7])
    ax_txt.axis("off")
    txt = ax_txt.text(
        0.01,
        0.99,
        "",
        va="top",
        ha="left",
        fontsize=9.5,
        linespacing=1.35,
        family="monospace",
        transform=ax_txt.transAxes,
    )

    hist = {
        k: deque([0.0] * HISTORY, maxlen=HISTORY)
        for k in (
            "cut_score",
            "struct_corr",
            "fade",
            "flicker",
            "core",
            "maps",
            "render",
        )
    }
    cut_frames = []
    cut_lines = []

    plt.ion()
    plt.show()
    fidx = 0
    sid, gid = 0, 0  # last-known shot id / group id (persist through drops)
    last_render = 0.0
    gc.disable()  # GC pauses are the main per-frame latency spike; reap manually below
    try:
        while plt.fignum_exists(fig.number):
            ret, frame = cap.read()
            if not ret:
                break
            fidx += 1
            if fidx % 300 == 0:
                gc.collect()  # bounded manual reap so memory stays in check

            t0 = time.perf_counter()
            fs, sig = gate.frame(frame)  # core pipeline
            t1 = time.perf_counter()
            motion = (
                fs.motion if fs.motion is not None else np.zeros((g, g), np.float32)
            )
            _ = (
                fs.saliency,
                fs.text,
                fs.focus,
                fs.structure_type,
                motion,
                fs.grid_V,
                fs.grid_S,  # force lazy maps
                fs.exposure,
                fs.contrast,
                fs.colorfulness,
                fs.detail,  # + scalars
                fs.flat_fraction,
                fs.noise_floor,
                fs.clipping,
            )
            t2 = time.perf_counter()
            t_core, t_maps = (t1 - t0) * 1e3, (t2 - t1) * 1e3

            if not (fs.blank or sig.freeze):  # blank/frozen frames are not shots
                sid, gid = tracker.update(fs, sig, fidx)

            if sig.cut:
                cut_frames.append(fidx - 1)
            for k, v in (
                ("cut_score", sig.cut_score),
                ("struct_corr", sig.struct_corr),
                ("fade", sig.fade),
                ("flicker", sig.flicker),
                ("core", t_core),
                ("maps", t_core + t_maps),
                ("render", last_render),
            ):
                hist[k].append(v)

            if fidx % DRAW_EVERY:  # compute every frame, draw every Nth
                continue

            # --- frame + banner ---
            im_frame.set_data(
                cv2.cvtColor(cv2.resize(frame, (W, H)), cv2.COLOR_BGR2RGB)
            )
            banner.set_text(
                "BLANK"
                if fs.blank
                else "CUT" if sig.cut else "FREEZE" if sig.freeze else ""
            )

            # --- maps (fixed clim) + grids (autoscaled) ---
            im_mot.set_data(motion)
            im_sal.set_data(fs.saliency)
            im_tex.set_data(fs.text)
            im_foc.set_data(fs.focus)
            im_st.set_data(fs.structure_type)
            for im, d in (
                (im_luma, fs.grid_V[:, :, 0]),
                (im_var, fs.grid_V[:, :, 1]),
                (im_sat, fs.grid_S[:, :, 0]),
            ):
                im.set_data(d)
                im.set_clim(float(d.min()), float(d.max()) + 1e-6)

            # --- cut timeline + markers ---
            cs = np.asarray(hist["cut_score"])
            ln_cut.set_ydata(cs)
            ax_cut.set_ylim(0, max(cs.max(), gate.cfg.cut_dissim) * 1.1 + 1e-3)
            for c in cut_lines:
                c.remove()
            cut_lines = []
            origin = fidx - HISTORY + 1
            for f in cut_frames:
                if origin <= f <= fidx:
                    cut_lines.append(
                        ax_cut.axvline(f - origin, color="red", lw=1.0, alpha=0.7)
                    )

            for ln, k in zip(ln_ev, ("struct_corr", "fade", "flicker")):
                ln.set_ydata(hist[k])
            for ln, k in zip(ln_lat, ("core", "maps")):
                ln.set_ydata(hist[k])
            a_core = np.mean(list(hist["core"])[-LAT_WIN:])
            a_full = np.mean(
                list(hist["maps"])[-LAT_WIN:]
            )  # maps history holds core+maps
            ax_lat.set_ylim(0, max(hist["maps"]) * 1.3 + 0.1)

            state = (
                "BLANK"
                if fs.blank
                else ("CUT" if sig.cut else ("FREEZE" if sig.freeze else "active"))
            )
            txt.set_text(
                f"state      {state}\n"
                f"shot       s{sid:<4d} group g{gid}\n"
                f"compute    {a_full:5.2f} ms   ({1000 / max(a_full, 1e-6):4.0f} fps)\n"
                f"  core     {a_core:5.2f} ms   + maps {a_full - a_core:4.2f}\n"
                f"  render   {last_render:5.2f} ms   (matplotlib, 1/{DRAW_EVERY} frames)\n"
                f"cut_score  {sig.cut_score:5.3f}   corr {sig.struct_corr:+.3f}\n"
                f"gain/bias  {sig.gain:5.2f} / {sig.bias:+.1f}\n"
                f"fade/flick {sig.fade:+.2f} / {sig.flicker:.2f}\n"
                f"exposure   {fs.exposure:5.1f}   contrast {fs.contrast:5.1f}\n"
                f"colorful   {fs.colorfulness:5.1f}   detail   {fs.detail:5.2f}\n"
                f"noise/clip {fs.noise_floor:5.2f} / {fs.clipping:+.2f}\n"
                f"flat_frac  {fs.flat_fraction:5.2f}   oriented {fs.orientedness:.2f}"
            )

            fig.suptitle(
                f"framegate   |   frame {fidx}   |   shot {sid} \u00b7 group {gid}"
                f"   |   {src}",
                fontsize=11,
            )
            tr = time.perf_counter()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_render = (time.perf_counter() - tr) * 1e3
    finally:
        gc.enable()
        cap.release()
        plt.ioff()
        print(f"done. {fidx} frames.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/visualize.py <video.mp4>")
        sys.exit(1)
    run(sys.argv[1])
