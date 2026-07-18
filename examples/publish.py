"""Drive framegate as a publishing gate: emit a Packet only for frames worth
processing (blank / frozen / duplicate frames are dropped) and print the inferred
stream metadata -- frame_id (over all inputs, so gaps are visible) and shot_id
(bumps on each cut). A pure consumer of the public API.

    python examples/publish.py path/to/video.mp4

With no argument it runs a short synthetic sequence (two shots, a blank gap, a held
frame) so every drop path is exercised without needing a video file.
"""

import sys

import cv2
import numpy as np

from framegate import Publisher


def video_frames(src):
    cap = cv2.VideoCapture(src)
    assert cap.isOpened(), f"cannot open {src}"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def synthetic_frames():
    """Shot A then shot B back-to-back (a direct cut -> shot_id bumps), with a held
    frame inside A and a blank gap after B -- so the run shows a cut plus the freeze
    and blank drop paths. (A cut is not detected *across* a blank, which resets the
    stream, so the cut is placed at the direct A->B boundary.)"""
    rng = np.random.default_rng(0)

    def shot(hue, n):
        base = (rng.random((128, 128)) * 120 + 80).astype(np.uint8)  # fixed luma layout
        out = []
        for _ in range(n):
            noise = rng.integers(-3, 4, base.shape)
            luma = np.clip(base + noise, 0, 255).astype(np.uint8)
            hsv = cv2.merge([np.full_like(luma, hue), np.full_like(luma, 200), luma])
            out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        return out

    a = shot(60, 12)
    a.insert(6, a[5])  # a held (byte-identical) frame -> freeze/duplicate drop
    b = shot(120, 10)  # different luma + hue, back-to-back -> a cut at the A/B boundary
    gap = [np.zeros((128, 128, 3), np.uint8)] * 3  # blank drop
    return a + b + gap


def main(frames):
    pub = Publisher()
    pub.subscribe(
        lambda p: print(
            f"  publish  frame_id={p.frame_id:4d}  shot_id={p.shot_id}"
            f"{'   <cut>' if p.signals.cut else ''}"
        )
    )
    published = dropped = total = 0
    for frame in frames:
        total += 1
        if pub.publish(frame) is None:
            dropped += 1
        else:
            published += 1
    print(
        f"\n{total} frames: {published} published, "
        f"{dropped} dropped (blank/frozen/duplicate)"
    )


if __name__ == "__main__":
    main(video_frames(sys.argv[1]) if len(sys.argv) == 2 else synthetic_frames())
