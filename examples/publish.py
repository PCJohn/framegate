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
    """Cross-cut A, B, A back-to-back (two direct cuts -> shot_id 0,1,2), where the
    two A shots share a layout so the third shot re-identifies as A's group
    (shot_group_id 0,1,0). A held frame inside the first A and a trailing blank gap
    exercise the freeze and blank drop paths too."""
    rng = np.random.default_rng(0)
    base_a = (rng.random((128, 128)) * 120 + 80).astype(np.uint8)  # A's fixed layout
    base_b = (rng.random((128, 128)) * 120 + 80).astype(np.uint8)  # B's fixed layout

    def shot(base, hue, n):
        out = []
        for _ in range(n):
            luma = np.clip(base + rng.integers(-3, 4, base.shape), 0, 255).astype(
                np.uint8
            )
            hsv = cv2.merge([np.full_like(luma, hue), np.full_like(luma, 200), luma])
            out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        return out

    a1 = shot(base_a, 60, 12)
    a1.insert(6, a1[5])  # a held (byte-identical) frame -> freeze/duplicate drop
    b = shot(base_b, 120, 10)  # different layout + hue -> a cut at each boundary
    a2 = shot(base_a, 60, 10)  # A returns -> should re-identify as group 0
    gap = [np.zeros((128, 128, 3), np.uint8)] * 3  # blank drop
    return a1 + b + a2 + gap


def main(frames):
    pub = Publisher()
    pub.subscribe(
        lambda p: print(
            f"  publish  frame_id={p.frame_id:4d}  shot_id={p.shot_id}"
            f"  shot_group_id={p.shot_group_id}"
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
