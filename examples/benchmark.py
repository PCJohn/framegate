"""Latency benchmark for framegate. Pure consumer of the public API; no timing
code lives in the library itself.

    python examples/benchmark.py            # synthetic 1080p frames
    python examples/benchmark.py video.mp4  # real video
"""

import sys
import time

import cv2
import numpy as np

from framegate import Gate


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


def _synthetic(n=300, h=1080, w=1920):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _time(fn, frames, warmup=10):
    for f in frames[:warmup]:
        fn(f)
    t = time.perf_counter()
    for f in frames:
        fn(f)
    return (time.perf_counter() - t) / len(frames) * 1e3


def main():
    if len(sys.argv) == 2:
        frames = _frames_from_video(sys.argv[1])
        label = sys.argv[1]
    else:
        frames = _synthetic()
        label = f"synthetic {frames[0].shape[1]}x{frames[0].shape[0]}"
    print(f"benchmark on {len(frames)} frames ({label})\n")

    g0 = Gate()
    ms_img = _time(lambda f: g0.image(f), frames)
    print(f"  image() (stateless)        {ms_img:6.2f} ms/frame  ({1000 / ms_img:5.0f} fps)")

    g = Gate()
    ms_vid = _time(lambda f: g.frame(f), frames)
    print(f"  frame() (temporal)         {ms_vid:6.2f} ms/frame  ({1000 / ms_vid:5.0f} fps)")

    # cost of also reading the feature maps (lazy; not paid unless accessed)
    gm = Gate()
    def _with_maps(f):
        fs, _ = gm.frame(f)
        fs.saliency; fs.fine_texture; _ = fs.motion
    ms_map = _time(_with_maps, frames)
    print(f"  frame() + saliency/motion  {ms_map:6.2f} ms/frame  ({1000 / ms_map:5.0f} fps)")

    # duplicate-heavy workload (every frame repeated) to show the lossless skip
    dup = [f for f in frames[:150] for _ in range(2)]
    g2 = Gate()
    ms_dup = _time(lambda f: g2.frame(f), dup)
    print(f"  frame() on 50% duplicates  {ms_dup:6.2f} ms/frame  ({1000 / ms_dup:5.0f} fps)")


if __name__ == "__main__":
    main()
