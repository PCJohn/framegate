# framegate

A fast, generic **pre-pipeline gate** for vision inputs. Run it first on any image
or video frame to get cheap, broadly-useful signals, then let a heavy model
(detector, OCR, VLM, encoder) run only **where** and **when** it's worth it.

It is deliberately *generic*: no task-specific heuristics, no per-dataset tuning.
Everything is derived from cheap per-cell colour/luma statistics computed on a small
thumbnail, so a single call costs ~1-2 ms on a 1080p frame.

```python
from framegate import Gate

gate = Gate()

# single image
stats = gate.image(img)
if not stats.blank:
    run_detector(img, saliency=stats.saliency)   # (G,G) maps; threshold/cluster as you like

# video
for frame in frames:
    stats, sig = gate.frame(frame)
    if sig.cut:
        start_new_shot()
    if not (stats.blank or sig.freeze):
        run_detector(frame, saliency=stats.saliency, motion=stats.motion)
```

## Install

`framegate` depends on [`tensorstats`](https://github.com/PCJohn/tensorstats) for the
fast moment computation. Install it first (it is not on PyPI):

```bash
pip install git+https://github.com/PCJohn/tensorstats
pip install framegate            # add [viz] for the example visualizer, [dev] for tests
```

## Concepts

The design has exactly two layers, split along the only axis that matters —
**whether a signal needs history**:

- **`FrameGate` / `FrameStats`** — stateless per-frame extraction. Works identically
  on a still image or a video frame. Produces all single-frame signals.
- **`StreamAnalyzer` / `TemporalSignals`** — the temporal layer. Consumes a stream of
  `FrameStats` and adds cut / freeze / fade / flicker.

`Gate` is a thin facade over both: `gate.image()` uses only the stateless layer;
`gate.frame()` runs both. A single image simply never touches the temporal layer —
that's the "use whichever parts apply" behaviour, for free.

Signals are **lazy properties**: you pay only for the ones you read. Reading
`stats.exposure` costs nothing extra; the saliency/fine_texture/motion arrays are computed
only on access.

## Signals

Single-frame (`FrameStats`, available for images and video):

| Signal           | Meaning |
|------------------|---------|
| `blank`          | Frame is empty/solid (skip everything downstream). |
| `exposure`       | Mean brightness (V). |
| `contrast`       | Brightness spread (V std). |
| `colorfulness`   | Mean saturation; ~0 for grayscale / line art. |
| `detail`         | Spatial complexity (mean cell variance; SI-like). |
| `flat_fraction`  | Fraction of near-uniform cells. |
| `clipping`       | Exposure asymmetry: >0 crushed shadows, <0 blown highlights. |
| `noise_floor`    | Std of the flattest cell (~sensor/compression noise). |
| `saliency`       | (G×G) coarse saliency map. |
| `fine_texture`   | (G×G) fine high-frequency achromatic texture; a generic text/print/UI cue (not OCR). |
| `motion`         | (G×G) motion magnitude vs the previous frame, or `None` for a still image (video only). |

Temporal (`TemporalSignals`, video only):

| Signal        | Meaning |
|---------------|---------|
| `cut`         | Shot boundary on this frame; `cut_frame` is its true index. |
| `cut_score`   | Raw dissimilarity that drives the cut decision. |
| `freeze`      | Frame is (near) identical to the previous one. |
| `fade`        | Signed fade strength, -1 (to black) .. +1 (from black). |
| `flicker`     | Periodic-brightness power fraction, 0..1 (strobing). |
| `struct_corr` | Motion-compensated luma correlation with the previous frame. |
| `gain`,`bias` | Global photometric change (a·prev+b) between frames. |

### How cut detection works

A cut is the **max of two complementary, motion-robust** dissimilarities, so either
can trigger it:

1. `1 - ` motion-compensated luma correlation — catches structural/spatial cuts and
   tolerates camera pans, but is blind to flat colour changes.
2. a normalized global saturation + hue-vector shift — catches equiluminant / flat
   colour cuts that the structural path misses.

A cut fires only when that score is a **robust (median+MAD) outlier**, is an
**isolated peak** (rejecting gradual pans/dissolves), and clears a **minimum shot
length** debounce — confirmed with one frame of latency.

### Feature maps: you build the regions

The gate returns the coarse per-cell maps and leaves region extraction to you — what counts
as a region (threshold, connected components, top-k cells, your own model on the cell stats)
is application-specific, so the library stays a pure, fast descriptor rather than baking in
one opinionated ROI policy:

- **`saliency`** — `(G×G)` appearance saliency: z-scored V-variance (texture) + S-mean
  (colorfulness) + center-surround luma contrast (each cell vs its local neighborhood,
  `sal_surround`), averaged and clipped at 0. Purely per-frame.
- **`fine_texture`** — `(G×G)` fine high-frequency achromatic texture, a generic
  text/print/UI cue (not OCR).
- **`motion`** — `(G×G)` motion magnitude vs the previous frame (`None` for a still image).
  See below.

All three are `(G×G)` (`G = 2**grid_exp`, default 32) at thumbnail scale; multiply cell
indices by `shape / G` to map back to source pixels. `saliency` and `fine_texture` are
cached on the `FrameStats`, so a duplicate frame reuses them for free; `motion` is recomputed
since it depends on the previous frame.

### Motion map and its local noise floor

`motion = |residual|`, where the residual is the per-cell luma change after fitting and
removing a global gain/bias (`a·prev + b`) — so a uniform brightness shift or auto-exposure
step does **not** read as motion. Raw, the residual still carries per-frame sensor/compression
speckle, so a **local** noise floor is subtracted:
`motion = max(|residual| - k·local_mean|residual|, 0)`, where `local_mean` is a box average
over a `motion_surround` neighborhood and `k = motion_floor_k` (default 1.0). Because the floor
is local rather than one global level, it adapts to regionally-varying noise (a busy/compressed
patch gets a higher threshold than a clean sky). Set `motion_floor_k = 0` for the raw magnitude.
The signed residual is also available as `stats.residual` if you want it before the absolute
value and threshold.

Note the trade this local (center-surround) floor makes: it favors motion **boundaries** over
filled interiors, so a large, smoothly-moving region is partly hollowed (its interior matches
its surround → near zero) while its edges remain, and it does **not** remove isolated single-cell
speckle. It is strongest when noise is regionally structured and motion is sparse; for a filled
region map, set `motion_floor_k = 0` and threshold `stats.residual` yourself.

### Scope: generic descriptors, not object detectors

framegate deliberately stops at low-level, semantic-free descriptors (the moment grids
and cheap combinations of them: contrast, colorfulness, texture energy, saliency). It
does **not** ship face / barcode / QR / logo detectors, and `fine_texture` is named for
the *pattern* it measures, not for "text" — it's one example of composing primitives
(high fine-contrast + achromatic + horizontal coherence), nothing more.

That boundary is intentional. Object-specific detection is either better served by a
real detector (a Haar/own cascade, a barcode/QR decoder, a small CNN) or is exactly the
thin model a *user* of this library builds on top of these statistics — "this
combination of patch statistics means a face for my data" is a downstream decision, not
a universal one. Baking objects in would trade the library's generality and tiny
footprint for heuristics that underperform the real thing. The classical literature
agrees: text/region localization is built from generic low-level features (edge density,
texture, colour) feeding heuristics or a tiny classifier — which is the layer above this
one.

## Configuration

All tunables live in one immutable `GateConfig`. Three ways to set them:

```python
from framegate import Gate, GateConfig

Gate()                                            # library defaults
Gate(GateConfig(min_scene_len=6, thumb=96))       # override in code
Gate(GateConfig.from_yaml("my_config.yaml"))      # load from a file
Gate(GateConfig.from_yaml("my_config.yaml", thumb=96))   # file + code overrides
```

`GateConfig` is the single source of truth for every tunable (see the dataclass for what
each does). There is no shipped YAML to drift from it: `from_yaml()` with no path returns
the defaults, and `GateConfig.to_yaml()` generates a template from the live fields on
demand (`python -m framegate > my_config.yaml`). Unknown keys raise immediately.

## Input formats

- **Colour**: BGR `uint8`, shape `(H, W, 3)` (OpenCV's native order).
- **Grayscale**: `uint8`, shape `(H, W)` or `(H, W, 1)`. Treated as `H=S=0, V=luma`,
  so colour signals correctly read as zero and cuts still fire via the luma path.

Any resolution is accepted; the frame is thumbnailed internally and map cells are mapped
back to source pixels.

### Reused frames

The gate already resizes the input and converts it to HSV. With `return_frames`
(default on), it hands those back on `FrameStats.thumb` (the resized BGR — or
single-channel grayscale — input) and `FrameStats.hsv`, both at `cfg.thumb`
resolution, so a driver doing `read → gate → heavy_pipeline` can reuse them instead
of recomputing. They are `None` when the flag is off.

## Output format

`FrameStats` and `TemporalSignals` are plain dataclasses holding the raw moment
arrays plus lazy properties; passing them around copies nothing. The one unavoidable
per-frame copy is the float64→float32 cast of the moment grids — if `tensorstats`
gains a float32 output mode, that copy disappears.

## Video-level optimization (lossless only)

By design, no optimization changes any output:

- **Blank → skip FAST.** A statistically flat frame has no corners, so the corner
  check is skipped. Bit-exact.
- **Duplicate → reuse stats.** Byte-identical consecutive frames reuse the previous
  result (identical input ⇒ identical stats), gated by a cheap strided pre-check so
  distinct frames pay only microseconds. Toggle with `skip_duplicates`.

  *Caveat:* this holds a reference to the previous frame. Sources that decode into one
  reused buffer **in place** would defeat it (OpenCV/imageio/decord all return fresh
  arrays and are safe); otherwise pass copies or set `skip_duplicates=False`.

Near-duplicate / signature-based skipping is intentionally **not** included because it
is not lossless.

One further skip is **effectively** (not provably bit-exact) lossless and on by
default: `fast_static` skips the sub-cell motion search when the zero-shift luma
correlation is already ≥ `static_corr` (0.98). On such near-static frames the search
cannot change the cut decision, and the reported `cut_score` differs by at most
`1 - static_corr` on those frames only. It saves ~0.2 ms on low-motion content and is
inert on high-motion frames; set `fast_static=False` for strict bit-exactness.

## Performance

Per-frame, single thread, 1080p (your mileage varies with content and hardware):

| Path                                          | Latency | Throughput |
|-----------------------------------------------|---------|------------|
| `image()` (stateless)                         | ~1.1 ms | ~900 fps   |
| `frame()`, high-motion (no skip)              | ~1.9 ms | ~520 fps   |
| `frame()`, low-motion (`fast_static` fires)   | ~1.7 ms | ~585 fps   |
| `+ saliency`/`motion` when read               | +0.5 ms |            |

(Measured on a slow CI-like box; a recent laptop is ~2–3× faster — e.g. `image()`
~0.55 ms, `frame()` ~1.2 ms.) The heavy pixel work (moments, FAST) is native; the rest
is small-array numpy. Buffers are preallocated and reused, work is float32, the rolling
baseline uses a sort instead of `np.median`, derived maps/ROIs are cached, and all array
outputs are lazy.

## Project layout

```
framegate/
├── pyproject.toml
├── README.md
├── src/framegate/
│   ├── __init__.py        # public API: Gate, GateConfig, FrameGate, FrameStats, ...
│   ├── config.py          # GateConfig (defaults, YAML, overrides)
│   ├── signals.py         # pure numeric core (no state, easy to test/port)
│   ├── stats.py           # FrameGate + FrameStats (per-frame extraction)
│   ├── stream.py          # StreamAnalyzer + TemporalSignals (temporal layer)
│   └── gate.py            # Gate facade + lossless duplicate skip
├── examples/
│   ├── visualize.py       # live matplotlib dashboard (frame + maps + signals)
│   └── benchmark.py       # latency measurement
└── tests/
    ├── synth.py           # synthetic scene builders
    ├── test_accuracy.py   # cuts, blank, freeze, fade, flicker, grayscale, maps, dedup
    ├── test_config.py     # YAML/dataclass parity, overrides, immutability
    └── test_latency.py    # per-frame latency budget
```

All visualization and timing code lives in `examples/` — the library itself contains
no matplotlib or benchmarking code.

## Examples & tests

```bash
python examples/visualize.py path/to/video.mp4     # needs [viz]
python examples/benchmark.py                        # synthetic, or pass a video path
pytest                                              # needs [dev]; -s prints latencies
```

The examples and tests import `framegate` exactly as a downstream user would.
