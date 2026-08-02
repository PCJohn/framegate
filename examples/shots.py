"""Live shot re-identification viewer: watch shot groups build up as the video plays,
so you can eyeball that the memory pulls up the right earlier shot on a recurrence.
Pure consumer of the public API (nothing here is imported by the library).

    python examples/shots.py path/to/video.mp4
    python examples/shots.py                    # no arg -> synthetic A,B,C,A,B demo
    python examples/shots.py video.mp4 --batch  # old whole-clip summary instead

Live layout (three panels):

  * left, large: the video playing, with a banner (frame / shot id / group id) that
    flashes on a cut and, on a re-identification, names the earlier shot it matched.
  * top right: the re-ID pair -- the current shot's first frame beside the last frame
    of the previous occurrence of that group. This is the check: the two should be the
    same setup. It holds on screen for a moment after each hit so you can look.
  * bottom right: the group gallery -- one thumbnail per group discovered so far,
    border-coloured, the active group boxed. The memory filling up, visible.

A re-identification is read straight off the public (shot_id, group_id) stream: when a
new shot reuses a group_id already seen, that is a recurrence, and the matched
reference is the most recent shot recorded for that group.

Requires the [viz] extra:  pip install "framegate[viz]"
"""

import gc
import math
import sys
import threading

import cv2
import matplotlib.pyplot as plt
import numpy as np

from framegate import Gate, ShotTracker

THUMB_W = 160  # display thumbnail width (px)
REP_OFFSET = 4  # representative frame = this many frames into a shot (skip transition)
HOLD = 24  # frames to keep a re-ID pair on screen after it fires
GALLERY = 12  # group thumbnails shown in the gallery (most recent groups)
DRAW_EVERY = 2  # redraw every N kept frames; compute still runs on all of them


def _thumb(frame):
    h = max(1, int(frame.shape[0] * THUMB_W / frame.shape[1]))
    return cv2.cvtColor(cv2.resize(frame, (THUMB_W, h)), cv2.COLOR_BGR2RGB)


def _color(g):
    return plt.get_cmap("tab20")(g % 20)


def _blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def _border(ax, color, lw=4):
    for sp in ax.spines.values():
        sp.set_color(color)
        sp.set_linewidth(lw)


class Groups:
    """The example's own bookkeeping over the public (shot_id, group_id) stream. Holds
    one representative thumbnail per group -- the clean exemplar the memory's accumulated
    bit-distribution effectively matches against -- so the re-ID panel can show it
    without the library knowing about display."""

    def __init__(self):
        self.rep = {}  # group_id -> representative thumbnail (settled first occurrence)
        self.first_sid = {}  # group_id -> shot_id where the group was first opened
        self.order = []  # group_ids in first-seen order

    def on_shot_open(self, gid, sid, first_thumb):
        """A shot just opened. Returns (reference thumbnail, reference shot_id) if this
        is a recurrence -- the group's representative and the shot that established it --
        else None. The memory matches the group's whole bit distribution, so its
        representative is the honest thing to show, not a boundary frame."""
        if gid in self.rep:  # recurrence
            return self.rep[gid], self.first_sid[gid]
        self.rep[gid] = first_thumb
        self.first_sid[gid] = sid
        self.order.append(gid)
        return None

    def on_frame(self, gid, thumb, n_in_shot):
        if n_in_shot == REP_OFFSET:  # settle the rep past the transition frame
            self.rep[gid] = thumb


def run_live(frames, cfg=None):
    _closed = threading.Event()
    gate = Gate(cfg)
    tracker = ShotTracker(gate.cfg)
    groups = Groups()

    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.7, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.18,
        wspace=0.12,
        left=0.02,
        right=0.98,
        top=0.9,
        bottom=0.03,
    )

    ax_vid = fig.add_subplot(gs[:, 0])
    ax_vid.axis("off")
    im_vid = ax_vid.imshow(np.zeros((2, 2, 3), np.uint8), aspect="auto")
    banner = ax_vid.text(
        0.5,
        0.95,
        "",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="white",
        transform=ax_vid.transAxes,
        bbox=dict(boxstyle="round", fc="black", alpha=0.6),
    )

    # re-ID pair: current shot (left) vs matched reference (right)
    gs_pair = gs[0, 1].subgridspec(1, 2, wspace=0.06)
    ax_cur = fig.add_subplot(gs_pair[0])
    ax_ref = fig.add_subplot(gs_pair[1])
    for ax in (ax_cur, ax_ref):
        _blank(ax)
    ax_cur.set_title("current shot", fontsize=9)
    ax_ref.set_title("matched reference", fontsize=9)
    im_cur = ax_cur.imshow(np.zeros((2, 2, 3), np.uint8), aspect="auto")
    im_ref = ax_ref.imshow(np.zeros((2, 2, 3), np.uint8), aspect="auto")

    # group gallery
    ax_gal = fig.add_subplot(gs[1, 1])
    ax_gal.axis("off")
    ax_gal.set_title("groups seen", fontsize=9)

    fig.suptitle("framegate shot re-identification", fontsize=13)

    plt.ion()
    fig.canvas.mpl_connect("close_event", lambda _e: _closed.set())
    plt.show()

    sid, gid = 0, 0
    n_in_shot = 0
    hold = 0  # frames left to keep the current re-ID pair up
    drew_gallery = -1  # last group count the gallery was drawn at
    fidx = -1
    for frame in frames:
        if _closed.is_set() or not plt.fignum_exists(fig.number):
            break
        fidx += 1
        if fidx % 300 == 0:
            gc.collect()  # bounded manual reap; matplotlib/numpy churn otherwise grows
        fs, sig = gate.frame(frame)
        if fs.blank or sig.freeze:
            continue

        new_sid, new_gid = tracker.update(fs, sig, fidx)
        disp = _thumb(frame)
        opened = new_sid != sid or fidx == 0
        if opened:
            sid, gid, n_in_shot = new_sid, new_gid, 0
            hit = groups.on_shot_open(gid, sid, disp)
            if hit is not None:  # recurrence: show the pair, flash the banner
                ref_thumb, ref_sid = hit
                im_cur.set_data(disp)
                im_ref.set_data(ref_thumb)
                for ax in (ax_cur, ax_ref):
                    _border(ax, _color(gid))
                ax_ref.set_title(f"group {gid} · first seen shot {ref_sid}", fontsize=9)
                hold = HOLD
        else:
            gid = new_gid  # stable within a shot; keep latest
        n_in_shot += 1
        groups.on_frame(gid, disp, n_in_shot)

        if fidx % DRAW_EVERY:
            continue

        im_vid.set_data(disp)
        recur = hold > 0
        banner.set_text(
            f"frame {fidx}   shot {sid}   group {gid}" + ("   ● RE-ID" if recur else "")
        )
        banner.set_color("#ffd54a" if recur else "white")
        _border(ax_vid, _color(gid), lw=6)
        if hold > 0:
            hold -= 1
        elif hold == 0:  # fade the pair's borders once the hold expires
            for ax in (ax_cur, ax_ref):
                _border(ax, "none", lw=0)
            hold = -1

        if len(groups.order) != drew_gallery:  # redraw gallery only when it grows
            _draw_gallery(fig, ax_gal, groups)
            drew_gallery = len(groups.order)

        if not _pump(fig):  # closed mid-frame -> stop cleanly
            break

    if not _closed.is_set() and plt.fignum_exists(fig.number):
        banner.set_text(f"done · {fidx + 1} frames · {len(groups.order)} groups")
        banner.set_color("white")
        _pump(fig)
        plt.ioff()
        plt.show(block=True)  # hold the final frame until the user closes it


def _pump(fig) -> bool:
    """Redraw and process UI events. Returns False if the window was closed during the
    call, so the caller can stop without the backend raising on a dead canvas."""
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
        return plt.fignum_exists(fig.number)
    except Exception:  # backend tears the canvas down mid-call on close
        return False


def _draw_gallery(fig, ax_gal, groups):
    """Repaint the group gallery: one representative thumbnail per group, newest last."""
    for old in list(ax_gal.figure.axes):
        if getattr(old, "_gallery", False):
            old.remove()
    shown = groups.order[-GALLERY:]
    cols = min(4, len(shown)) or 1
    rows = max(1, math.ceil(len(shown) / cols))
    box = ax_gal.get_position()
    cw, ch = box.width / cols, box.height / rows
    for i, g in enumerate(shown):
        r, c = divmod(i, cols)
        sub = fig.add_axes(
            [box.x0 + c * cw, box.y0 + box.height - (r + 1) * ch, cw * 0.92, ch * 0.82]
        )
        sub._gallery = True
        _blank(sub)
        sub.imshow(groups.rep[g])
        _border(sub, _color(g), lw=3)
        sub.set_title(f"g{g}", fontsize=8, color=_color(g), pad=1)


# --- batch summary (the old behaviour, kept for a quick whole-clip overview) ---


def run_batch(frames, cfg=None):
    gate = Gate(cfg)
    tracker = ShotTracker(gate.cfg)
    shots, cur = [], None
    for fidx, frame in enumerate(frames):
        fs, sig = gate.frame(frame)
        if fs.blank or sig.freeze:
            continue
        sid, gid = tracker.update(fs, sig, fidx)
        if cur is None or sid != cur["sid"]:
            if cur is not None:
                shots.append(cur)
            cur = dict(sid=sid, gid=gid, first_id=fidx, rep=_thumb(frame), n=1)
        else:
            cur["gid"], cur["n"] = gid, cur["n"] + 1
            if cur["n"] == REP_OFFSET:
                cur["rep"] = _thumb(frame)
    if cur is not None:
        shots.append(cur)

    n_groups = len({s["gid"] for s in shots})
    print(f"{len(shots)} shots, {n_groups} groups")
    for s in shots:
        print(f"  shot {s['sid']:3d}  group {s['gid']:3d}  first frame {s['first_id']}")

    disp = shots[:48]
    cols = min(8, len(disp)) or 1
    rows = max(1, math.ceil(len(disp) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 1.9, rows * 2.1), squeeze=False
    )
    fig.suptitle(f"{len(shots)} shots  |  {n_groups} groups", fontsize=12)
    for i, ax in enumerate(axes.flat):
        _blank(ax)
        if i >= len(disp):
            ax.axis("off")
            continue
        s = disp[i]
        ax.imshow(s["rep"])
        _border(ax, _color(s["gid"]))
        ax.set_title(f"s{s['sid']} · g{s['gid']}", fontsize=9, color=_color(s["gid"]))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def video_frames(path):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"cannot open {path}"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def synthetic_frames():
    """A,B,C,A,B: three setups, two recurring -> groups 0,1,2,0,1 with two re-ID hits."""
    rng = np.random.default_rng(0)
    bases = {k: (rng.random((128, 128)) * 120 + 80).astype(np.uint8) for k in "ABC"}

    def shot(key, hue, n=20):
        out = []
        for _ in range(n):
            luma = np.clip(bases[key] + rng.integers(-3, 4, (128, 128)), 0, 255)
            luma = luma.astype(np.uint8)
            hsv = cv2.merge([np.full_like(luma, hue), np.full_like(luma, 200), luma])
            out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        return out

    return (
        shot("A", 60) + shot("B", 120) + shot("C", 30) + shot("A", 60) + shot("B", 120)
    )


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    batch = "--batch" in argv[1:]
    frames = video_frames(args[0]) if args else synthetic_frames()
    (run_batch if batch else run_live)(frames)


if __name__ == "__main__":
    main(sys.argv)
