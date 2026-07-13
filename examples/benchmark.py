"""Latency benchmark for framegate. A pure consumer of the public API -- no timing
code lives in the library itself.

    python examples/benchmark.py             # synthetic sweep (sizes / grids / strides)
    python examples/benchmark.py video.mp4   # measure on a real video

Methodology: minimum over repeats with the GC disabled, and sweep configs are
*interleaved* (round-robin one pass each per round) so thermal/clock drift over the
run is spread evenly across configs rather than penalizing whichever ran during a
throttle. Median is shown too: a large min/median gap flags a noisy machine.

These are small-array ops, so OpenCV's default multithreading can add scheduling
variance at this size; if your sweep numbers look noisy, `cv2.setNumThreads(1)` (or
profiling your real workload) usually steadies them. The library never sets this
globally -- a library shouldn't mutate process-wide state. Absolute numbers scale
with CPU clock (battery vs. charger, throttling)."""

import gc
import platform
import sys
import time

import cv2
import numpy as np

from framegate import Gate, GateConfig


def _bench_group(items, warmup=10, repeats=7):
    """items: list of (label, fn, frames). Returns {label: (min_ms, median_ms)}.
    Configs are interleaved across repeat rounds to spread drift."""
    for _, fn, frames in items:
        for f in frames[:warmup]:
            fn(f)
    samples = {label: [] for label, _, _ in items}
    gc.disable()
    try:
        for _ in range(repeats):
            for label, fn, frames in items:
                t = time.perf_counter()
                for f in frames:
                    fn(f)
                samples[label].append((time.perf_counter() - t) / len(frames) * 1e3)
    finally:
        gc.enable()
    return {label: (min(v), float(np.median(v))) for label, v in samples.items()}


def _synthetic(n, h, w, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _frames_from_video(path, n=300):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"cannot open {path}"
    out = []
    while len(out) < n:
        ret, f = cap.read()
        if not ret:
            break
        out.append(f)
    cap.release()
    return out


def _read_maps(g):
    names = (
        "saliency",
        "text",
        "focus",
        "structure_type",
        "edge_energy",
        "coherence",
        "cornerness",
        "orientation",
        "motion",
    )

    def f(frame):
        fs, _ = g.frame(frame)
        for n in names:
            _ = getattr(fs, n)

    return f


def _print(label, mn_md):
    mn, md = mn_md
    print(f"  {label:36s} {mn:6.3f} / {md:6.3f} ms   {1000 / mn:6.0f} fps")


def _header(title):
    print(f"\n{title}\n  {'config':36s} {'min / median':>14s}   {'(from min)':>10s}")


def _cfg_items(specs, frames):
    """Build (label, image-fn, frames) items."""
    items = []
    for label, cfg in specs:
        try:
            g = Gate(cfg)
        except (RuntimeError, ValueError) as e:
            print(f"  {label:36s} unsupported ({str(e).split(':')[-1].strip()})")
            continue
        items.append((label, (lambda gg: lambda f: gg.image(f))(g), frames))
    return items


RESOLUTIONS = [
    ("360p", 360, 640),
    ("480p", 480, 854),
    ("720p", 720, 1280),
    ("1080p", 1080, 1920),
    ("1440p", 1440, 2560),
    ("2160p/4K", 2160, 3840),
]


def run_synthetic():
    print("framegate benchmark")
    print(
        f"  platform: {platform.system()} {platform.machine()} | python {platform.python_version()} "
        f"| numpy {np.__version__} | cv2 {cv2.__version__} (threads={cv2.getNumThreads()})"
    )
    print("  timing: min / median ms per frame, GC off, sweeps interleaved\n")

    frames = _synthetic(120, 1080, 1920)

    _header("[1] 1080p summary (default config)")
    gi, gv, gm, gd = Gate(), Gate(), Gate(), Gate()
    dup = [x for x in frames[:60] for _ in range(2)]
    res = _bench_group(
        [
            ("image()  (stateless)", lambda f: gi.image(f), frames),
            ("frame()  (temporal)", lambda f: gv.frame(f), frames),
            ("frame()  + all maps read", _read_maps(gm), frames),
            ("frame()  on 50% duplicates", lambda f: gd.frame(f), dup),
        ]
    )
    for label in [
        "image()  (stateless)",
        "frame()  (temporal)",
        "frame()  + all maps read",
        "frame()  on 50% duplicates",
    ]:
        _print(label, res[label])

    _header("[2] input-size sweep (default config: grid 32, 4-level pyramid, stride 2)")
    # measured per-resolution (frames freed between) so 4K doesn't blow up memory; the
    # cross-resolution signal is large, so sequential measurement is fine here
    for name, h, w in RESOLUTIONS:
        fr = _synthetic(60 if h >= 1440 else 120, h, w)
        gi2, gf2 = Gate(), Gate()
        r = _bench_group(
            [
                ("image", lambda f, g=gi2: g.image(f), fr),
                ("frame", lambda f, g=gf2: g.frame(f), fr),
            ]
        )
        mi, mf = r["image"][0], r["frame"][0]
        print(
            f"  {name:10s} {w}x{h:<6d}  image {mi:6.3f} ms ({1000/mi:5.0f} fps)   "
            f"frame {mf:6.3f} ms ({1000/mf:5.0f} fps)"
        )
        del fr

    _header(
        "[3] grid-size sweep (1080p, default stride 2)  -- grid = 2**grid_exp cells/dim"
    )
    res = _bench_group(
        _cfg_items(
            [
                (f"grid_exp={ge}  ({2**ge}x{2**ge} cells)", GateConfig(grid_exp=ge))
                for ge in (4, 5, 6)
            ],
            frames,
        )
    )
    for label in res:
        _print(label, res[label])

    _header("[4] stride sweep (1080p, grid 32)  -- imfeat pixel subsampling")
    res = _bench_group(
        _cfg_items(
            [(f"stride={st}", GateConfig(stride=st)) for st in (1, 2, 3, 4)], frames
        )
    )
    for label in res:
        _print(label, res[label])

    _header("[5] thumb-size sweep (1080p, grid 32, default stride 2)")
    res = _bench_group(
        _cfg_items(
            [(f"thumb={tb}", GateConfig(thumb=tb)) for tb in (64, 96, 128, 192, 256)],
            frames,
        )
    )
    for label in res:
        _print(label, res[label])

    _header(
        "[6] pyramid-depth sweep (1080p, grid 32, stride 2)  -- n_levels = grids per pass"
    )
    res = _bench_group(
        _cfg_items(
            [
                (f"n_levels={nl}  (32..{2**(5 - nl + 1)})", GateConfig(n_levels=nl))
                for nl in (1, 2, 3, 4)
            ],
            frames,
        )
    )
    for label in res:
        _print(label, res[label])


def run_video(path):
    frames = _frames_from_video(path)
    print(f"framegate benchmark on {len(frames)} frames from {path}")
    print(
        f"  resolution {frames[0].shape[1]}x{frames[0].shape[0]} "
        f"| cv2 threads={cv2.getNumThreads()} | timing min / median, GC off\n"
    )
    gi, gv, gm = Gate(), Gate(), Gate()
    res = _bench_group(
        [
            ("image()  (stateless)", lambda f: gi.image(f), frames),
            ("frame()  (temporal)", lambda f: gv.frame(f), frames),
            ("frame()  + all maps read", _read_maps(gm), frames),
        ]
    )
    for label in [
        "image()  (stateless)",
        "frame()  (temporal)",
        "frame()  + all maps read",
    ]:
        _print(label, res[label])


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_video(sys.argv[1])
    else:
        run_synthetic()
