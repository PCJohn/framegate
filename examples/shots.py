"""Shot analyzer: segment a video into shots and re-identify recurring ones, then
show the result for visual verification. Pure consumer of the public API.

    python examples/shots.py path/to/video.mp4
    python examples/shots.py                      # no arg -> synthetic A,B,C,A,B demo

Two matplotlib windows open:

  1. Shot filmstrip -- one representative frame per shot, in order, border-coloured by
     shot_group_id. Recurrences share a colour, so cross-cutting (A,B,A,B...) is
     obvious at a glance.
  2. Re-ID verification -- for every shot that re-identified as an earlier group, the
     new shot's first frame (left) beside the reference frame it matched, i.e. the
     previous occurrence's last frame (right), with the match score `z`. Eyeball these
     to confirm the grouping is right (or catch a false merge).

This is the batch companion to visualize.py's live dashboard: same idioms (cv2 +
matplotlib, no library deps), but a whole-clip summary rather than a per-frame view.

Requires the [viz] extra:  pip install "framegate[viz]"
"""

import math
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from framegate import Gate, ShotTracker

THUMB_W = 160  # thumbnail width for display (px)
MAX_SHOTS = 48  # cap the filmstrip so a long clip stays readable
MAX_HITS = 8  # cap the verification panel
REP_OFFSET = 4  # representative = this many frames into the shot (skip the transition)
PROGRESS_EVERY = 100  # frames between progress prints (batch runs on long clips)


def _thumb(frame):
    h = max(1, int(frame.shape[0] * THUMB_W / frame.shape[1]))
    return cv2.cvtColor(cv2.resize(frame, (THUMB_W, h)), cv2.COLOR_BGR2RGB)


def analyze(frames, cfg=None, progress=True):
    """Run the gate + shot tracker over `frames` (an iterable of BGR arrays). Returns
    (gate, shots) where each shot is a dict with its id, group, frame range, first and
    representative thumbnails, and first-frame pHash (for re-ID verification). Processes
    the whole clip up front (it is a batch tool), printing progress to stderr; only two
    thumbnails are kept per shot, so memory stays flat on long videos."""
    gate = Gate(cfg)
    tracker = ShotTracker(gate.cfg)
    shots, cur = [], None
    fidx = -1
    for fidx, frame in enumerate(frames):
        if progress and fidx % PROGRESS_EVERY == 0:
            print(
                f"\r  analyzing... frame {fidx}, {len(shots)} shots",
                end="",
                file=sys.stderr,
                flush=True,
            )
        fs, sig = gate.frame(frame)
        if fs.blank or sig.freeze:  # not a shot -> skip (as Publisher would drop it)
            continue
        sid, gid = tracker.update(fs, sig, fidx)
        if cur is None or sid != cur["sid"]:
            if cur is not None:
                shots.append(cur)
            th = _thumb(frame)  # one resize per shot start, not per frame
            cur = dict(
                sid=sid,
                gid=gid,
                first_id=fidx,
                last_id=fidx,
                n=1,
                first=th,
                rep=th,
                first_hash=int(fs.phash),
            )
        else:
            cur["gid"] = gid  # stable within a shot; keep the latest
            cur["last_id"] = fidx
            cur["n"] += 1
            if cur["n"] == REP_OFFSET:  # a settled frame stands in for the shot
                cur["rep"] = _thumb(frame)
    if cur is not None:
        shots.append(cur)
    if progress:
        print(
            f"\r  analyzed {fidx + 1} frames -> {len(shots)} shots" + " " * 12,
            file=sys.stderr,
        )
    return gate, shots


def reid_hits(gate, shots):
    """Shots that re-identified as an already-seen group, each paired with the earlier
    occurrence and the Hamming distance between their first-frame pHashes (the signal
    the memory now matches on). We compare first frames: a shot's first frame is
    latency-clean, whereas its last frame is the next shot's first frame bleeding in
    (cut is confirmed one frame late)."""
    last_of, hits = {}, []
    for s in shots:
        prev = last_of.get(s["gid"])
        if prev is not None:
            d = bin(s["first_hash"] ^ prev["first_hash"]).count("1")
            hits.append((s, prev, d))
        last_of[s["gid"]] = s
    return hits


def _paint_border(ax, color, lw=4):
    for sp in ax.spines.values():
        sp.set_color(color)
        sp.set_linewidth(lw)


def show(shots, hits):
    cmap = plt.get_cmap("tab20")

    def color(g):
        return cmap(g % 20)

    n_groups = len({s["gid"] for s in shots})

    # --- window 1: shot filmstrip, border-coloured by group ---
    disp = shots[:MAX_SHOTS]
    cols = min(8, len(disp)) or 1
    rows = max(1, math.ceil(len(disp) / cols))
    fig1, axes = plt.subplots(
        rows, cols, figsize=(cols * 1.9, rows * 2.1), squeeze=False
    )
    extra = f"   (showing first {MAX_SHOTS})" if len(shots) > MAX_SHOTS else ""
    fig1.suptitle(f"{len(shots)} shots   |   {n_groups} groups{extra}", fontsize=12)
    for i, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(disp):
            ax.axis("off")
            continue
        s = disp[i]
        ax.imshow(s["rep"])
        _paint_border(ax, color(s["gid"]))
        ax.set_title(f"s{s['sid']}  ·  g{s['gid']}", fontsize=9, color=color(s["gid"]))
    fig1.tight_layout(rect=(0, 0, 1, 0.95))

    # --- window 2: re-ID verification pairs ---
    if not hits:
        fig2, ax = plt.subplots(figsize=(6, 1.6))
        ax.axis("off")
        ax.text(0.5, 0.5, "no recurring shots detected", ha="center", va="center")
    else:
        hd = hits[:MAX_HITS]
        fig2, axes = plt.subplots(
            len(hd), 2, figsize=(5.2, 2.3 * len(hd)), squeeze=False
        )
        fig2.suptitle(
            "re-ID: new shot (left)  vs  matched reference (right)", fontsize=11
        )
        for r, (s, prev, z) in enumerate(hd):
            la, ra = axes[r]
            la.imshow(s["first"])
            ra.imshow(prev["first"])
            for ax in (la, ra):
                ax.set_xticks([])
                ax.set_yticks([])
                _paint_border(ax, color(s["gid"]))
            la.set_title(f"s{s['sid']} first  (frame {s['first_id']})", fontsize=8)
            ra.set_title(
                f"\u2190 s{prev['sid']} (frame {prev['first_id']})   d={z}bits",
                fontsize=8,
            )
            la.set_ylabel(f"group {s['gid']}", fontsize=9, color=color(s["gid"]))
        fig2.tight_layout(rect=(0, 0, 1, 0.96))

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
    """A,B,C,A,B: three setups, two of them recurring -> groups 0,1,2,0,1 with two
    re-ID hits, so both windows have something to show without needing a video."""
    rng = np.random.default_rng(0)
    bases = {k: (rng.random((128, 128)) * 120 + 80).astype(np.uint8) for k in "ABC"}

    def shot(key, hue, n=14):
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
    frames = video_frames(argv[1]) if len(argv) > 1 else synthetic_frames()
    gate, shots = analyze(frames)
    hits = reid_hits(gate, shots)

    n_groups = len({s["gid"] for s in shots})
    print(f"{len(shots)} shots, {n_groups} groups")
    for i, s in enumerate(shots):
        end = shots[i + 1]["first_id"] - 1 if i + 1 < len(shots) else s["last_id"]
        print(
            f"  shot {s['sid']:3d}  group {s['gid']:3d}  frames {s['first_id']}..{end}"
        )
    if hits:
        print("re-ID hits:")
        for s, prev, z in hits:
            print(
                f"  shot {s['sid']} -> group {s['gid']} (shot {prev['sid']}, d={z}bits)"
            )
    show(shots, hits)


if __name__ == "__main__":
    main(sys.argv)
