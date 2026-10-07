"""framegate on a video with no dashboard: the loop of examples/visualize.py without the draw,
for timing and for profilers.

    python gate_loop.py video.mp4 [--threads 2] [--model path.fdt] [--model-threads 4] [--frames 600] [--no-maps]
    python gate_loop.py video.mp4 --threads 2 --model m.fdt --pyspy gate.json [--pyspy-seconds 30]

Prints the per-frame median of gate.frame(), of the lazy maps the dashboard reads, and --
with a model -- of the model stage alone (compose, then pack + score or score_maps, timed
inside the gate), with the thread count the scorer actually runs on, so ``--model-threads``
can be compared.
``--pyspy OUT`` attaches py-spy to this process *after* the gate is built and the first frame
has run, so the profile holds frames rather than the model's imports (fastdet's import pulls
in CatBoost, pandas and pyarrow, which took the whole 30 s of a ``py-spy record -- python``
run). py-spy must be on PATH. It records Python frames at 50 Hz, which py-spy keeps up with;
``--pyspy-native`` adds C/C++ frames, which on Windows makes py-spy fall behind ("behind in
sampling") and smears the timing, so use it only to see *which* native calls, not how long.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import time

import cv2
import numpy as np

from framegate import Gate, GateConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--model", action="append", default=[], metavar="[NAME=]PATH")
    ap.add_argument(
        "--model-threads",
        type=int,
        default=None,
        help="the scorer's own thread count (default: --threads); bit-identical at any count",
    )
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--no-maps", action="store_true", help="time gate.frame() only")
    ap.add_argument(
        "--packed",
        action="store_true",
        help="score through the packed matrix (pack_native + score) instead of the maps, "
        "for a live A/B on one build",
    )
    ap.add_argument(
        "--return-frames",
        action="store_true",
        help="keep GateConfig.return_frames on (the pass then writes the thumbnail out)",
    )
    ap.add_argument(
        "--pyspy",
        metavar="OUT",
        help="attach py-spy once the gate is warm; speedscope file",
    )
    ap.add_argument("--pyspy-seconds", type=int, default=30)
    ap.add_argument("--pyspy-rate", type=int, default=50)
    ap.add_argument(
        "--pyspy-native",
        action="store_true",
        help="also unwind C/C++ frames (slow on Windows)",
    )
    args = ap.parse_args()
    overrides = {}
    if args.threads:
        overrides["feat_threads"] = args.threads
    if args.model:
        overrides["models"] = {}
        for spec in args.model:
            name, _, path = spec.rpartition("=")
            overrides["models"][name or "text"] = path
    if not args.return_frames:
        overrides["return_frames"] = False
    if args.model_threads:
        overrides["model_threads"] = args.model_threads
    model_ms: list[float] = (
        []
    )  # per-frame time of the models' maps, timed inside the gate
    # ... and its parts: compose, then either pack + score (the packed matrix) or score_maps
    # (the scorer reading the maps where they lie, fastdet's score-maps commit and later)
    stage_ms: dict[str, list[float]] = {
        "compose": [],
        "pack": [],
        "score": [],
        "score_maps": [],
    }
    if args.model:
        from framegate import models as models_module

        original_maps = models_module.ModelBank.maps

        def timed_maps(self, result, shape):
            t0 = time.perf_counter()
            out = original_maps(self, result, shape)
            model_ms.append((time.perf_counter() - t0) * 1e3)
            return out

        models_module.ModelBank.maps = timed_maps  # type: ignore[method-assign]
        # the parts, where fastdet's classes expose them (compose -> pack -> score)
        try:
            import fastdet.detector as fd_detector
            import fastdet.features as fd_features
            import fastdet.native as fd_native

            def _timed(cls, name, key):
                original = getattr(cls, name)

                def wrapper(self, *a, **k):
                    t0 = time.perf_counter()
                    out = original(self, *a, **k)
                    stage_ms[key].append((time.perf_counter() - t0) * 1e3)
                    return out

                setattr(cls, name, wrapper)

            _timed(fd_features.FeatureExtractor, "compose", "compose")
            _timed(fd_detector.Detector, "pack_native", "pack")
            _timed(fd_native.NativeScorer, "score", "score")
            if hasattr(fd_native.NativeScorer, "score_maps"):
                _timed(fd_native.NativeScorer, "score_maps", "score_maps")
            if (
                args.packed
            ):  # the path before fastdet's score-maps commit, on this build

                def packed_predict(self, level_maps, broadcast_vecs):
                    native = self.pack_native(level_maps, broadcast_vecs)
                    return self.native.score(
                        native, use_exit=self.config.model.use_exit
                    ).reshape(64, 64)

                fd_detector.Detector._predict_from_maps = packed_predict  # type: ignore[method-assign]
        except (ImportError, AttributeError) as exc:  # another fastdet layout: no parts
            print(f"(model stage parts not timed: {exc})")
    gate = Gate(GateConfig(**overrides))
    models = list(gate.models)  # (close() forgets them)
    scorer_threads = dict(getattr(gate, "model_threads", {}))
    cap = cv2.VideoCapture(args.video)
    core, maps = [], []
    spy = None
    gc.disable()
    try:
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                break
            if (
                i == 1 and args.pyspy
            ):  # the gate is built and warm: profile frames, not imports
                cmd = [
                    "py-spy",
                    "record",
                    "--pid",
                    str(os.getpid()),
                    "--rate",
                    str(args.pyspy_rate),
                    "--duration",
                    str(args.pyspy_seconds),
                    "--format",
                    "speedscope",
                    "-o",
                    args.pyspy,
                ]
                if args.pyspy_native:
                    cmd.insert(3, "--native")
                spy = subprocess.Popen(cmd)
                time.sleep(1.0)  # let it attach
            t0 = time.perf_counter()
            fs, _sig = gate.frame(frame)
            t1 = time.perf_counter()
            if not args.no_maps:
                _ = (
                    fs.saliency,
                    fs.text,
                    fs.focus,
                    fs.structure_type,
                    fs.grid_V,
                    fs.grid_S,
                    fs.exposure,
                    fs.contrast,
                    fs.colorfulness,
                    fs.detail,
                    fs.flat_fraction,
                    fs.noise_floor,
                    fs.clipping,
                )
            t2 = time.perf_counter()
            core.append(t1 - t0)
            maps.append(t2 - t1)
    finally:
        gc.enable()
        if spy is not None:
            spy.wait()  # py-spy writes its file when its duration is up
        gate.close()
    core_ms, maps_ms = 1e3 * np.array(core), 1e3 * np.array(maps)
    print(
        f"{len(core)} frames, {gate.cfg.feat_threads} imfeat thread(s), models {models or '-'}"
    )
    print(
        f"gate.frame()   median {np.median(core_ms):.2f} ms  p90 {np.percentile(core_ms, 90):.2f}  min {core_ms.min():.2f}"
    )
    if model_ms:
        m = np.array(model_ms)
        threads = ", ".join(f"{n}: {t}" for n, t in scorer_threads.items()) or "?"
        print(
            f"  model stage  median {np.median(m):.2f} ms  p90 {np.percentile(m, 90):.2f}  (scorer threads {threads})"
        )
        parts = {k: np.array(v) for k, v in stage_ms.items() if v}
        if parts:
            print(
                "    of which   "
                + "   ".join(f"{k} {np.median(v):.2f} ms" for k, v in parts.items())
                + "   (medians; the rest is Python between them)"
            )
    if not args.no_maps:
        print(f"+ lazy maps    median {np.median(maps_ms):.2f} ms")


if __name__ == "__main__":
    main()
