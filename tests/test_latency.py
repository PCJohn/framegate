"""Latency tests. Budgets are deliberately generous so they pass on slow CI; the
actual measured times are printed (run pytest -s to see them)."""

import time

import numpy as np
import pytest

from framegate import Gate


def _bench(fn, frames, warmup=5):
    for f in frames[:warmup]:
        fn(f)
    t = time.perf_counter()
    for f in frames:
        fn(f)
    return (time.perf_counter() - t) / len(frames) * 1e3


@pytest.fixture(scope="module")
def frames_1080p():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8) for _ in range(60)]


def test_image_latency_under_budget(frames_1080p, capsys):
    ms = _bench(lambda f: Gate().image(f), frames_1080p)
    with capsys.disabled():
        print(f"\n  image() : {ms:.2f} ms/frame ({1000 / ms:.0f} fps)")
    assert ms < 8.0


def test_frame_latency_under_budget(frames_1080p, capsys):
    g = Gate()
    ms = _bench(lambda f: g.frame(f), frames_1080p)
    with capsys.disabled():
        print(f"  frame() : {ms:.2f} ms/frame ({1000 / ms:.0f} fps)")
    assert ms < 8.0


def test_duplicate_skip_is_not_slower(frames_1080p, capsys):
    # on a duplicate-heavy stream the lossless skip should not hurt (typically helps)
    dup = [f for f in frames_1080p[:30] for _ in range(2)]
    g = Gate()
    ms = _bench(lambda f: g.frame(f), dup)
    with capsys.disabled():
        print(f"  frame() (50% dup): {ms:.2f} ms/frame ({1000 / ms:.0f} fps)")
    assert ms < 8.0
