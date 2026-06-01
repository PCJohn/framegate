"""Latency tests. Budgets are deliberately generous so they pass on slow CI; the
actual measured times are printed (run `pytest -s` to see them). Timing uses the
minimum over repeats with the GC disabled -- scheduling/GC jitter can only add time,
so the minimum is the least-perturbed estimate of intrinsic cost. The detailed
size/grid/stride sweep lives in examples/benchmark.py."""

import gc
import time

import numpy as np
import pytest

from framegate import Gate

BUDGET_MS = 8.0   # generous; real hardware is far under this


def _bench(fn, frames, warmup=8, repeats=5):
    for f in frames[:warmup]:
        fn(f)
    best = float("inf")
    gc.disable()
    try:
        for _ in range(repeats):
            t = time.perf_counter()
            for f in frames:
                fn(f)
            best = min(best, (time.perf_counter() - t) / len(frames) * 1e3)
    finally:
        gc.enable()
    return best


@pytest.fixture(scope="module")
def frames_1080p():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8) for _ in range(60)]


def test_image_latency_under_budget(frames_1080p, capsys):
    g = Gate()
    ms = _bench(lambda f: g.image(f), frames_1080p)
    with capsys.disabled():
        print(f"\n  image()              : {ms:.2f} ms/frame ({1000 / ms:.0f} fps)")
    assert ms < BUDGET_MS


def test_frame_latency_under_budget(frames_1080p, capsys):
    g = Gate()
    ms = _bench(lambda f: g.frame(f), frames_1080p)
    with capsys.disabled():
        print(f"  frame()              : {ms:.2f} ms/frame ({1000 / ms:.0f} fps)")
    assert ms < BUDGET_MS


def test_frame_with_maps_under_budget(frames_1080p, capsys):
    g = Gate()

    def read_maps(f):
        fs, _ = g.frame(f)
        fs.saliency; fs.fine_texture; _ = fs.motion

    ms = _bench(read_maps, frames_1080p)
    with capsys.disabled():
        print(f"  frame() + all maps   : {ms:.2f} ms/frame ({1000 / ms:.0f} fps)")
    assert ms < BUDGET_MS


def test_duplicate_skip_is_not_slower(frames_1080p, capsys):
    dup = [f for f in frames_1080p[:30] for _ in range(2)]   # every frame repeated
    g = Gate()
    ms = _bench(lambda f: g.frame(f), dup)
    with capsys.disabled():
        print(f"  frame() (50% dup)    : {ms:.2f} ms/frame ({1000 / ms:.0f} fps)")
    assert ms < BUDGET_MS
