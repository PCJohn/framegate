"""Latency benchmark for framegate. A pure consumer of the public API -- no timing
code lives in the library itself.

    python examples/benchmark.py             # synthetic sweep (sizes / grids / strides)
    python examples/benchmark.py video.mp4   # measure on a real video at 1080p-style settings

Timing methodology: minimum over repeats with the GC disabled. Scheduling and GC
pauses can only *add* time, so the minimum is the least-perturbed estimate of the
intrinsic cost; median is also shown so a large min/median gap flags a noisy machine.
Absolute numbers still scale with CPU clock (battery vs. charger, thermal throttling)."""

import gc
import platform
import sys
import time

import cv2
import numpy as np

from framegate import Gate, GateConfig


def _bench(fn, frames, warmup=10, repeats=7):
    for f in frames[:warmup]:
        fn(f)
    times = []
    gc.disable()
    try:
        for _ in range(repeats):
            t = time.perf_counter()
            for f in frames:
                fn(f)
            times.append((time.perf_counter() - t) / len(frames) * 1e3)
    finally:
        gc.enable()
    return min(times), float(np.median(times))


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


def _row(label, fn, frames):
    mn, md = _bench(fn, frames)
    print(f"  {label:34s} {mn:6.3f} / {md:6.3f} ms   {1000 / mn:6.0f} fps")


def _header(title):
    print(f"\n{title}\n  {'config':34s} {'min / median':>14s}   {'(from min)':>10s}")


def _read_maps(g):
    def f(frame):
        fs, _ = g.frame(frame)
        fs.saliency; fs.fine_texture; _ = fs.motion
    return f


RESOLUTIONS = [("360p", 360, 640), ("480p", 480, 854), ("720p", 720, 1280),
               ("1080p", 1080, 1920), ("1440p", 1440, 2560), ("2160p/4K", 2160, 3840)]


def _row_cfg(label, cfg, frames):
    try:
        g_ = Gate(cfg)
    except RuntimeError as e:
        print(f"  {label:34s} unsupported ({str(e).split(':')[-1].strip()})")
        return
    _row(label, lambda f: g_.image(f), frames)


def run_synthetic():
    cv2_threads = cv2.getNumThreads()
    print("framegate benchmark")
    print(f"  platform: {platform.system()} {platform.machine()} | python {platform.python_version()} "
          f"| numpy {np.__version__} | cv2 {cv2.__version__} (threads={cv2_threads})")
    print("  timing: min / median ms per frame over repeats, GC disabled\n")

    frames = _synthetic(120, 1080, 1920)
    _header("[1] 1080p summary (default config)")
    g = Gate(); _row("image()  (stateless)", lambda f: g.image(f), frames)
    gv = Gate(); _row("frame()  (temporal)", lambda f: gv.frame(f), frames)
    gm = Gate(); _row("frame()  + all maps read", _read_maps(gm), frames)
    dup = [x for x in frames[:60] for _ in range(2)]
    gd = Gate(); _row("frame()  on 50% duplicates", lambda f: gd.frame(f), dup)

    _header("[2] input-size sweep (default grid 32, stride 1)")
    for name, h, w in RESOLUTIONS:
        n = 120 if h <= 1080 else 60
        fr = _synthetic(n, h, w)
        gi = Gate(); gf = Gate()
        mi, _ = _bench(lambda f: gi.image(f), fr)
        mf, _ = _bench(lambda f: gf.frame(f), fr)
        print(f"  {name:10s} {w}x{h:<6d}  image {mi:6.3f} ms ({1000/mi:5.0f} fps)   "
              f"frame {mf:6.3f} ms ({1000/mf:5.0f} fps)")

    _header("[3] grid-size sweep (1080p, stride 1)  -- grid = 2**grid_exp cells/dim")
    for ge in (4, 5, 6):     # grid_exp 7 (128x128) exceeds tensorstats' int16 cell limit
        _row_cfg(f"grid_exp={ge}  ({2**ge}x{2**ge} cells)  image()", GateConfig(grid_exp=ge), frames)

    _header("[4] stride sweep (1080p, grid 32)  -- tensorstats subsampling")
    for st in (1, 2, 3, 4):
        _row_cfg(f"stride={st}  image()", GateConfig(stride=st), frames)

    _header("[5] thumb-size sweep (1080p, grid 32, stride 1)")
    for tb in (64, 96, 128, 192, 256):
        _row_cfg(f"thumb={tb}  image()", GateConfig(thumb=tb), frames)


def run_video(path):
    frames = _frames_from_video(path)
    print(f"framegate benchmark on {len(frames)} frames from {path}")
    print(f"  resolution {frames[0].shape[1]}x{frames[0].shape[0]} | timing min / median, GC off\n")
    g = Gate(); _row("image()  (stateless)", lambda f: g.image(f), frames)
    gv = Gate(); _row("frame()  (temporal)", lambda f: gv.frame(f), frames)
    gm = Gate(); _row("frame()  + all maps read", _read_maps(gm), frames)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_video(sys.argv[1])
    else:
        run_synthetic()
