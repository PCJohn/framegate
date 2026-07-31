"""Latency benchmark for framegate. A pure consumer of the public API -- no timing
code lives in the library itself.

    python examples/benchmark.py             # synthetic sweep (sizes / grids / strides)
    python examples/benchmark.py video.mp4   # measure on a real video

Sections [1] and [6]-[10] sweep whole-call latency across configs. [2]-[5] break a
frame open: where the time goes inside one, what the lazy maps cost, and how imfeat's
worker pool and OpenCV's interact.

[2] and [3] measure things the sweeps cannot see. [2] is cumulative -- each row is a
real measurement including everything above it, so the deltas are differences of
measurements rather than of separately-minimised timings. It reads private attributes
and prints a skip line if they move. [3] exists because FrameStats properties are lazy:
frame() builds the object without evaluating a single map, so every other number here
excludes them. Its rows are timed on an uncached FrameStats, so shared intermediates are
paid for by whichever property is measured first -- read the ranking, not the sum.

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


def _header(title, cols=True):
    print(f"\n{title}")
    if cols:
        print(f"  {'config':36s} {'min / median':>14s}   {'(from min)':>10s}")


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


def _phases(cfg, frames):
    """Cumulative nested timings, so each row is a real measurement and the deltas are
    differences of measurements rather than of separately-minimised sweeps. Reaches into
    private attributes; returns None if the internals have moved."""
    g = Gate(cfg)
    try:
        fg = g._gate
        fc, keep = fg._feat, fg.cfg.return_frames
        hsv = fg._to_hsv(frames[0], keep)[0]
        view = fc._view(hsv)
    except AttributeError:
        return None
    gf = Gate(cfg)
    return _bench_group(
        [
            ("thumb", lambda f: fg._to_hsv(f, keep), frames),
            (
                "+ imfeat core",
                lambda f: (fg._to_hsv(f, keep), fc._impl.features(view)),
                frames,
            ),
            (
                "+ python wrap",
                lambda f: (fg._to_hsv(f, keep), fc.features(hsv)),
                frames,
            ),
            ("+ temporal", lambda f: gf.frame(f), frames),
        ]
    )


MAP_PROPS = (
    "saliency",
    "text",
    "focus",
    "structure_type",
    "detail",
    "flat_fraction",
    "noise_floor",
    "edge_energy",
    "coherence",
    "cornerness",
    "motion",
    "grid_V",
    "exposure",
    "contrast",
    "colorfulness",
    "sharpness",
    "orientedness",
    "dominant_orientation",
    "clipping",
    "blank",
)


def _map_costs(frames, warmup=6, repeats=25):
    """Per-property cost on an *uncached* FrameStats. frame() evaluates none of these --
    they are lazy -- so nothing else in this file measures them."""
    g = Gate()
    for f in frames[:warmup]:
        g.frame(f)
    out = {}
    gc.disable()
    try:
        for name in MAP_PROPS:
            best = None
            for i in range(repeats):
                fs, _ = g.frame(frames[i % len(frames)])
                t = time.perf_counter()
                try:
                    getattr(fs, name)
                except (AttributeError, ValueError, TypeError):
                    best = None  # property gone or not available on this frame
                    break
                d = time.perf_counter() - t
                best = d if best is None else min(best, d)
            if best is not None:
                out[name] = best * 1e3
    finally:
        gc.enable()
    return out


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

    _header("[2] where a 1080p frame goes (cumulative; deltas are between rows)", False)
    ph = _phases(GateConfig(), frames)
    if ph is None:
        print("  (skipped: framegate internals have moved)")
    else:
        prev = 0.0
        for label in ("thumb", "+ imfeat core", "+ python wrap", "+ temporal"):
            mn = ph[label][0]
            print(
                f"  {label:36s} {mn:6.3f} / {ph[label][1]:6.3f} ms   delta {mn - prev:+6.3f}"
            )
            prev = mn

    _header("[3] lazy map cost (frame() evaluates none of these)", False)
    mc = _map_costs(frames)
    tot = sum(mc.values())
    for name, v in sorted(mc.items(), key=lambda kv: -kv[1])[:8]:
        bar = "#" * max(1, round(30 * v / max(mc.values())))
        print(f"  {name:36s} {v:6.3f} ms        {100 * v / tot:4.1f}%  {bar}")
    top = dict(sorted(mc.items(), key=lambda kv: -kv[1])[:8])
    rest = sum(v for n, v in mc.items() if n not in top)
    print(
        f"  {f'({len(mc) - len(top)} others)':36s} {rest:6.3f} ms        {100 * rest / tot:4.1f}%"
    )
    print(f"  {'sum -- double-counts shared work':36s} {tot:6.3f} ms")

    _header("[4] feat_threads sweep (1080p) -- imfeat worker threads")
    res = _bench_group(
        _cfg_items(
            [(f"feat_threads={n}", GateConfig(feat_threads=n)) for n in (1, 2, 3, 4)],
            frames,
        )
    )
    for label in res:
        _print(label, res[label])

    _header("[5] cv2 threads x feat_threads (1080p) -- pool contention", False)
    dflt = cv2.getNumThreads()
    hdr = "cv2 / feat"
    print(f"  {hdr:36s}" + "".join(f"{n:>10d}" for n in (1, 2, 4)))
    try:
        for nc in dict.fromkeys((1, 2, 4, dflt)):
            cv2.setNumThreads(nc)
            row = _bench_group(
                _cfg_items(
                    [(f"f{n}", GateConfig(feat_threads=n)) for n in (1, 2, 4)], frames
                )
            )
            tag = f"{nc}" + (" (default)" if nc == dflt else "")
            print(
                f"  {tag:36s}" + "".join(f"{row[f'f{n}'][0]:10.3f}" for n in (1, 2, 4))
            )
    finally:
        cv2.setNumThreads(dflt)

    _header("[6] input-size sweep (default config: grid 32, 4-level pyramid, stride 2)")
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
        "[7] grid-size sweep (1080p, default stride 2)  -- grid = 2**grid_exp cells/dim"
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

    _header("[8] stride sweep (1080p, grid 32)  -- imfeat pixel subsampling")
    res = _bench_group(
        _cfg_items(
            [(f"stride={st}", GateConfig(stride=st)) for st in (1, 2, 3, 4)], frames
        )
    )
    for label in res:
        _print(label, res[label])

    _header("[9] thumb-size sweep (1080p, grid 32, default stride 2)")
    res = _bench_group(
        _cfg_items(
            [(f"thumb={tb}", GateConfig(thumb=tb)) for tb in (64, 96, 128, 192, 256)],
            frames,
        )
    )
    for label in res:
        _print(label, res[label])

    _header(
        "[10] pyramid-depth sweep (1080p, grid 32, stride 2)  -- n_levels = grids per pass"
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

    _header("where a frame goes (cumulative; deltas are between rows)", False)
    ph = _phases(GateConfig(), frames)
    if ph is None:
        print("  (skipped: framegate internals have moved)")
    else:
        prev = 0.0
        for label in ("thumb", "+ imfeat core", "+ python wrap", "+ temporal"):
            mn = ph[label][0]
            print(
                f"  {label:36s} {mn:6.3f} / {ph[label][1]:6.3f} ms   delta {mn - prev:+6.3f}"
            )
            prev = mn

    _header("lazy map cost on this content", False)
    mc = _map_costs(frames)
    tot = sum(mc.values())
    for name, v in sorted(mc.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {name:36s} {v:6.3f} ms        {100 * v / tot:4.1f}%")
    print(f"  {'sum -- double-counts shared work':36s} {tot:6.3f} ms")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_video(sys.argv[1])
    else:
        run_synthetic()
