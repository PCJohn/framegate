"""Where a gate.frame() goes, stage by stage: the imfeat pass (frame in, thumbnail made
inside), the model stage, the rest of FrameGate.process() (the numpy conversions), and the
temporal layer after it.  Also the pass on a ready thumbnail (cv2.resize first), which
isolates the in-pass resize.

    python gate_stages.py video.mp4 [--threads 2] [--model m.fdt] [--frames 200]
"""

from __future__ import annotations

import argparse
import gc
import time

import cv2
import imfeat
import numpy as np

from framegate import Gate, GateConfig
from framegate import models as models_module
from framegate import stats as stats_module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()
    overrides = {"feat_threads": args.threads, "return_frames": False}
    if args.model:
        overrides["models"] = {}
        for spec in args.model:
            name, _, path = spec.rpartition("=")
            overrides["models"][name or "text"] = path

    # per frame: the stages' times, NaN where a stage did not run (a frame byte-identical to
    # the previous one is skipped by the gate -- GateConfig.skip_duplicates -- and runs nothing)
    stage: dict[str, list[float]] = {k: [] for k in ("pass", "models", "process")}
    frame_ms: list[float] = []

    def record(key, dt):
        stage[key][-1] = dt

    orig_features = imfeat.FeatureComputer.features

    def timed_features(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_features(self, *a, **k)
        record("pass", time.perf_counter() - t0)
        return out

    imfeat.FeatureComputer.features = timed_features  # type: ignore[method-assign]

    orig_maps = models_module.ModelBank.maps

    def timed_maps(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_maps(self, *a, **k)
        record("models", time.perf_counter() - t0)
        return out

    models_module.ModelBank.maps = timed_maps  # type: ignore[method-assign]

    orig_process = stats_module.FrameGate.process

    def timed_process(self, frame):
        t0 = time.perf_counter()
        out = orig_process(self, frame)
        record("process", time.perf_counter() - t0)
        return out

    stats_module.FrameGate.process = timed_process  # type: ignore[method-assign]

    gate = Gate(GateConfig(**overrides))
    models = list(gate.models)  # (close() forgets them)
    cap = cv2.VideoCapture(args.video)
    gc.disable()
    frames = []
    try:
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                break
            for v in stage.values():
                v.append(np.nan)
            t0 = time.perf_counter()
            gate.frame(frame)
            frame_ms.append(time.perf_counter() - t0)
            if i < 8:
                frames.append(frame.copy())
    finally:
        gc.enable()
    n = len(frame_ms)
    if not frames:
        raise SystemExit(f"no frames read from {args.video}")
    # the ready-thumbnail pass, same operating point (threads, grid, stride), cv2 resizing
    cfg = gate.cfg
    size = cfg.thumb_hw(frames[0].shape)
    feat = imfeat.FeatureComputer(
        shape=(*size, 3),
        grid=[(e, e) for e in cfg.pyramid_exps],
        stride=cfg.stride_for(size),
        threads=cfg.feat_threads,
        input_space="bgr",
        feature_space="hsv",
        thumb=None,
    )
    thumb = np.empty((*size, 3), np.uint8)
    thumb_pass, cv2_resize = [], []
    for _ in range(3):
        for f in frames:
            cv2.resize(f, (size[1], size[0]), dst=thumb, interpolation=cv2.INTER_AREA)
            orig_features(feat, thumb)
    for _ in range(max(1, n // len(frames))):
        for f in frames:
            t0 = time.perf_counter()
            cv2.resize(f, (size[1], size[0]), dst=thumb, interpolation=cv2.INTER_AREA)
            t1 = time.perf_counter()
            orig_features(feat, thumb)
            t2 = time.perf_counter()
            cv2_resize.append(t1 - t0)
            thumb_pass.append(t2 - t1)
    gate.close()
    ms = {k: 1e3 * np.array(v) for k, v in stage.items()}
    frame_arr = 1e3 * np.array(frame_ms)
    processed = ~np.isnan(ms["process"])
    skipped = int((~processed).sum())
    med = np.nanmedian
    print(
        f"{n} frames of {frames[0].shape[1]}x{frames[0].shape[0]} -> thumbnail {size[1]}x{size[0]},"
        f" {cfg.feat_threads} imfeat thread(s), models {models or '-'}"
        + (
            f"; {skipped} duplicate frames skipped by the gate (medians over the rest)"
            if skipped
            else ""
        )
    )
    print(
        f"gate.frame()            median {med(frame_arr[processed]):.2f} ms"
        + (f"   (all frames {np.median(frame_arr):.2f})" if skipped else "")
    )
    print(f"  imfeat pass (fused)   median {med(ms['pass']):.2f} ms")
    model_part = np.nan_to_num(ms["models"], nan=0.0) if models else np.zeros(n)
    if models:
        print(f"  model stage           median {med(ms['models']):.2f} ms")
    rest = ms["process"] - np.nan_to_num(ms["pass"], nan=0.0) - model_part
    print(
        f"  process() python rest median {med(rest):.2f} ms  (conversions, struct copies, FrameStats)"
    )
    after = frame_arr - ms["process"]
    print(f"  temporal layer after  median {med(after):.2f} ms")
    thumb_arr, resize_arr = 1e3 * np.array(thumb_pass), 1e3 * np.array(cv2_resize)
    print(
        f"ready-thumbnail pass    median {np.median(thumb_arr):.2f} ms  (cv2.resize INTER_AREA alongside"
        f" {np.median(resize_arr):.2f} ms, cv2 threads {cv2.getNumThreads()})"
    )
    print(f"  => the in-pass resize ~ {med(ms['pass']) - np.median(thumb_arr):.2f} ms")


if __name__ == "__main__":
    main()
