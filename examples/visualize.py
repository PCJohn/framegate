"""Live dashboard for framegate on a video. Pure consumer of the public API
(nothing here is imported by the library itself).

    python examples/visualize.py path/to/video.mp4
    python examples/visualize.py clip.mp4 --threads 4     # imfeat worker threads

Shows the appearance maps (motion / saliency / text / focus / structure), the per-cell
moment grids, the temporal event signals (cut / fade / flicker / struct-corr),
and a live latency panel separating framegate compute from matplotlib render --
so the speed of the package is visible against the cost of just drawing it.

Keys: ``q``/``Esc`` quit, ``space`` pause, ``s`` save the figure to the working
directory. Requires the [viz] extra:  pip install "framegate[viz]"
"""

import argparse
import gc
import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import theme

from framegate import Gate, GateConfig, ShotTracker

HISTORY = 300  # rolling time-series window (viz only)
DRAW_EVERY = 3  # redraw the dashboard every N frames; compute still runs every frame
DISPLAY_MAX = 480  # longest side of the displayed frame
# Fixed display ranges for the map panels, so a near-static frame stays dark
# instead of auto-stretching its noise floor to full brightness.
MAP_VMAX = {"motion": 32.0, "saliency": 3.0, "texture": 24.0, "focus": 30.0}


def run(src, cfg=None):
    cap = cv2.VideoCapture(src)
    assert cap.isOpened(), f"cannot open {src}"
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    scale = min(1.0, DISPLAY_MAX / max(sw, sh))
    W, H = max(1, int(sw * scale)), max(1, int(sh * scale))

    gate = Gate(cfg)
    g = gate.cfg.grid_size
    tracker = ShotTracker(gate.cfg)  # shot_id + shot_group_id

    theme.apply()
    fig = plt.figure(figsize=(17, 9), layout="constrained")
    gs = fig.add_gridspec(4, 7, hspace=0.08, wspace=0.05)

    ax_frame = fig.add_subplot(gs[0:2, 0:3])
    ax_frame.set_axis_off()
    im_frame = ax_frame.imshow(np.zeros((H, W, 3), np.uint8), aspect="auto")
    banner = ax_frame.text(
        0.5,
        0.93,
        "",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=theme.TEXT,
        transform=ax_frame.transAxes,
        bbox={"boxstyle": "round", "fc": theme.BG, "ec": theme.EDGE, "alpha": 0.75},
    )

    # appearance maps (fixed clim, no colorbars)
    im_mot = theme.heat_axis(
        fig.add_subplot(gs[0, 3]), "motion (illum-inv)", "hot", (0, MAP_VMAX["motion"])
    )
    im_sal = theme.heat_axis(
        fig.add_subplot(gs[0, 4]), "saliency", "magma", (0, MAP_VMAX["saliency"])
    )
    im_tex = theme.heat_axis(
        fig.add_subplot(gs[0, 5]), "text", "cividis", (0, MAP_VMAX["texture"])
    )
    # exact per-cell moment grids (autoscaled)
    im_luma = theme.heat_axis(fig.add_subplot(gs[1, 3]), "luma  (V mean)", "inferno")
    im_var = theme.heat_axis(fig.add_subplot(gs[1, 4]), "contrast  (V var)", "viridis")
    im_sat = theme.heat_axis(
        fig.add_subplot(gs[1, 5]), "saturation  (S mean)", "plasma"
    )
    im_foc = theme.heat_axis(
        fig.add_subplot(gs[0, 6]),
        "focus (edge sharpness)",
        "bone",
        (0, MAP_VMAX["focus"]),
    )
    ax_st = fig.add_subplot(gs[1, 6])
    ax_st.set_title("structure  (RGB = flat/edge/tex)")
    ax_st.set_xticks([])
    ax_st.set_yticks([])
    im_st = ax_st.imshow(
        np.zeros((2, 2, 3), np.float32), aspect="auto", interpolation="nearest"
    )

    # cut-score timeline with threshold + cut markers
    ax_cut = fig.add_subplot(gs[2, 0:3])
    (ln_cut,) = theme.series_axis(
        ax_cut, "cut score   (threshold dotted, cut = rose)", ["cut score"], HISTORY
    )
    ax_cut.axhline(gate.cfg.cut_dissim, color=theme.ROSE, ls=":", lw=1.0)

    # latency timeline: framegate compute vs matplotlib render
    ax_lat = fig.add_subplot(gs[2, 3:7])
    ln_lat = theme.series_axis(
        ax_lat,
        f"framegate latency   [{gate.cfg.feat_threads} imfeat thread(s)]",
        ["core (gate)", "core + maps"],
        HISTORY,
        [theme.AMBER, theme.GREEN],
    )
    ax_lat.set_ylabel("ms")
    ax_lat.set_xlabel(f"last {HISTORY} frames")

    # temporal event signals
    ax_ev = fig.add_subplot(gs[3, 0:2])
    ax_ev.set_ylim(-1.05, 1.05)
    ln_ev = theme.series_axis(
        ax_ev,
        "events",
        ["struct_corr", "fade", "flicker"],
        HISTORY,
        [theme.GREEN, theme.BLUE, theme.VIOLET],
    )

    txt = theme.readout(fig.add_subplot(gs[3, 2:7]), fontsize=9.5)

    hist = {  # NaN until a frame lands there: the plots and medians skip the gap
        k: deque([np.nan] * HISTORY, maxlen=HISTORY)
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

    window = theme.Window(fig, "framegate", save_prefix="framegate_dashboard")
    fidx = 0
    sid, gid = 0, 0  # last-known shot id / group id (persist through drops)
    last_render = 0.0
    fps = None
    last_tick = time.perf_counter()
    gc.disable()  # GC pauses are the main per-frame latency spike; reap manually below
    try:
        while not window.closed:
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
            now = time.perf_counter()
            instant = 1.0 / max(now - last_tick, 1e-6)
            fps = instant if fps is None else 0.9 * fps + 0.1 * instant
            last_tick = now

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
            ax_cut.set_ylim(
                0, max(float(np.nanmax(cs)), gate.cfg.cut_dissim) * 1.1 + 1e-3
            )
            for c in cut_lines:
                c.remove()
            cut_lines = []
            origin = fidx - HISTORY + 1
            for f in cut_frames:
                if origin <= f <= fidx:
                    cut_lines.append(
                        ax_cut.axvline(f - origin, color=theme.ROSE, lw=1.0, alpha=0.8)
                    )

            for ln, k in zip(ln_ev, ("struct_corr", "fade", "flicker"), strict=True):
                ln.set_ydata(hist[k])
            for ln, k in zip(ln_lat, ("core", "maps"), strict=True):
                ln.set_ydata(hist[k])
            med_core = float(np.nanmedian(hist["core"]))
            med_full = float(np.nanmedian(hist["maps"]))  # maps history holds core+maps
            top = float(np.nanpercentile(hist["maps"], 98)) * 1.2
            ax_lat.set_ylim(0, max(top, 1.0))

            state = (
                "BLANK"
                if fs.blank
                else ("CUT" if sig.cut else ("FREEZE" if sig.freeze else "active"))
            )
            txt.set_text(
                f"state      {state}\n"
                f"shot       s{sid:<4d} group g{gid}\n"
                f"core (gate)  {t_core:6.2f} ms   median {med_core:6.2f}\n"
                f"+ maps       {t_maps:6.2f} ms   median {med_full - med_core:6.2f}\n"
                f"render       {last_render:6.2f} ms   (matplotlib, 1/{DRAW_EVERY} frames)\n"
                f"total        {t_core + t_maps:6.2f} ms   {fps:5.1f} fps   frame {fidx}\n"
                f"cut_score  {sig.cut_score:5.3f}   corr {sig.struct_corr:+.3f}\n"
                f"gain/bias  {sig.gain:5.2f} / {sig.bias:+.1f}\n"
                f"fade/flick {sig.fade:+.2f} / {sig.flicker:.2f}\n"
                f"exposure   {fs.exposure:5.1f}   contrast {fs.contrast:5.1f}\n"
                f"colorful   {fs.colorfulness:5.1f}   detail   {fs.detail:5.2f}\n"
                f"noise/clip {fs.noise_floor:5.2f} / {fs.clipping:+.2f}\n"
                f"flat_frac  {fs.flat_fraction:5.2f}   oriented {fs.orientedness:.2f}"
            )

            fig.suptitle(
                f"framegate   |   frame {fidx}   |   shot {sid} · group {gid}"
                f"   |   {src}"
            )
            last_render = window.draw()
            window.pump()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        gc.enable()
        cap.release()
        window.close()
        plt.ioff()
        if fidx:
            print(
                f"done. {fidx} frames; gate median {np.nanmedian(hist['core']):.2f} ms,"
                f" with maps {np.nanmedian(hist['maps']):.2f} ms"
                f" ({gate.cfg.feat_threads} imfeat thread(s))."
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="live framegate dashboard on a video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="a video file (anything OpenCV can open)")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=f"imfeat worker threads (default {GateConfig().feat_threads}; "
        "the output is bit-identical at any count)",
    )
    args = parser.parse_args(argv)
    cfg = GateConfig(feat_threads=args.threads) if args.threads else None
    run(args.video, cfg)


if __name__ == "__main__":
    main()
