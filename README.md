# framegate

A fast, generic **pre-pipeline gate** for vision inputs. Run it first on any image
or video frame to get cheap, broadly-useful signals, then let a heavy model
(detector, OCR, VLM, encoder) run only **where** and **when** it's worth it.

It is deliberately *generic*: no task-specific heuristics, no per-dataset tuning.
Everything is derived from cheap per-cell colour/luma statistics computed on a small
thumbnail, so a single call costs about 1.5-2 ms on a 1080p frame on a modern laptop --
and tunes down to well under 1 ms when you need it (hardware-dependent; see Performance).

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

`framegate` depends on [`imfeat`](https://github.com/PCJohn/imfeat), which computes the
moments *and* the gradient structure-tensor features in a single pass over the pixels.
Install it first (it is not on PyPI):

```bash
pip install git+https://github.com/PCJohn/imfeat
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

For streaming, **`Publisher`** wraps `Gate` as a pub/sub-style node: feed it frames and
it emits a `Packet` — the frame plus its `FrameStats` (pyramid + maps), `TemporalSignals`,
and inferred stream metadata (`frame_id` over all inputs, `shot_id` that increments on each
cut, and `shot_group_id` shared by recurrences of the same shot — so a dialogue that
cross-cuts between three setups emits `shot_id` 0,1,2,3,4,5… with `shot_group_id`
0,1,2,0,1,2…) — but **only** for frames worth downstream work. Blank frames and freezes (a
byte- or near-identical duplicate surfaces as `freeze`) are dropped, so a subscriber can
assume every `Packet` is worth the heavy pipeline. Read the return of `publish(frame)`
(`None` when dropped) or register a callback via `subscribe`; the emit path is a plain
fan-out, so a real
transport (zmq, asyncio queue, ROS) can replace it later without touching the gate.

**Shot re-identification** (`shotmem.py`, `reid.py`) is what gives `shot_group_id` its
meaning, and it's a three-layer memory. **L1** (`stream.py`) is the freeze check: a frame
is a duplicate if it affine-matches a recent kept frame, so held frames are dropped before
any heavy work. **L2** is the shot store: each shot's first-frame luma pHash goes into a
[framestore](https://github.com/PCJohn/framestore) Hamming index, and at each cut the new
shot's first frame queries it for candidate groups within `reid_maxd` (relative Hamming —
the recall net). Each candidate is then scored by a per-bit Bernoulli model: a shot is a
distribution over hash bits, and a frame matches if its mean per-bit log-likelihood under
that distribution clears `reid_ll` (the precision dial). Bits that vary within a shot (a
moving mouth) sit near 0.5 and stop penalising, so the same setup re-IDs across pose and
speech changes — looser, and cheaper, than the old NCC heuristic. Per-frame cost is one
list append; scoring is integer until the final compare. **L3** (`longterm.py`) is a stub
for an offline-built, mmap-loaded prototype index (a later pass).

Signals are **lazy properties**: you pay only for the ones you read. Reading
`stats.exposure` costs nothing extra; the saliency/text/motion arrays are computed
only on access.

**Under the hood.** Every signal is derived from **exact per-cell central moments**
(mean, variance, and two higher moments) of the H/S/V channels, computed over a
`thumb`x`thumb` thumbnail in a single pass by
[`imfeat`](https://github.com/PCJohn/imfeat) -- which, in that same pass and on the same
cells, also computes the gradient structure tensor, an orientation histogram and extrema
densities, for *every* channel. The thumbnail is divided into a
`GxG` grid of cells (`G = 2**grid_exp`, default 32); each cell's moments summarize its
colour and texture, and the signals are cheap combinations of them. The grid is also
computed at several coarser resolutions in the *same* pass -- a dyadic **pyramid** of
`n_levels` (default 4: 32 -> 16 -> 8 -> 4). Today the signals read only the finest level;
the coarser levels are plumbing for upcoming multi-scale signals, so **set `n_levels=1`
to skip them** (~0.5 ms cheaper) until you need them.

## Recommended usage

The gate is a cheap front door: run it on every frame, then spend real compute only where
and when it says to.

1. **One `Gate` per stream, constructed once.** It owns the scratch buffers and the
   temporal state. Not thread-safe -- use one `Gate` per thread.
2. **Branch on the cheap signals before the expensive model:**
   - `stats.blank` -> skip the frame entirely.
   - `sig.cut` -> reset shot-level state (re-key a tracker, start a new segment).
   - `sig.freeze` -> reuse the previous heavy result; the frame didn't change.
   - otherwise -> run your model, optionally only on high-`saliency` / high-`motion` cells.
3. **Reuse the work the gate already did.** With `return_frames=True` (default) the resized
   thumbnail and HSV are on `stats.thumb` / `stats.hsv` -- pass them downstream instead of
   resizing again. Set it `False` if you don't, to save the copy.
4. **For steady latency, disable the GC around your loop.** GC pauses -- not framegate --
   are the main source of latency *spikes*; `gc.disable()` plus a periodic `gc.collect()`
   flattens the tail.
5. **Tune to your budget with `thumb` first, then `stride`** (see [Performance](#performance)).

```python
import gc
from framegate import Gate

gate = Gate()
gc.disable()
for i, frame in enumerate(stream):
    stats, sig = gate.frame(frame)
    if i % 500 == 0:
        gc.collect()                          # bounded manual reap
    if stats.blank or sig.freeze:
        continue                              # nothing worth running a model on
    if sig.cut:
        tracker.reset()                       # shot boundary
    regions = pick_cells(stats.saliency, stats.motion)   # your ROI policy
    run_model(stats.thumb, regions)           # reuse the thumbnail the gate made
```

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
| `text`           | (G×G) text likelihood: fine, achromatic, horizontally-coherent, bimodal texture. A cue for dense/printed text, not OCR. |
| `edge_energy`    | (G×G) gradient (edge) energy per cell. |
| `coherence`      | (G×G) edge anisotropy in [0,1]; 1 = a single dominant orientation. |
| `cornerness`     | (G×G) Shi-Tomasi corner strength (smaller structure-tensor eigenvalue). |
| `orientation`    | (G×G) dominant edge orientation in radians; reliability is `coherence`. |
| `sharpness`      | Scalar global gradient energy (`log1p`); raw detail/contrast. |
| `focus`          | (G×G) edge sharpness = energy / variance (≈1/edge-width²); contrast-independent focus/defocus map. |
| `structure_type` | (G×G×3) soft `[flat, edge, structured]` (sums to 1); `argmax` for a hard label. |
| `orientedness`   | Scalar global edge anisotropy [0,1]; high = one dominant frame-wide orientation. |
| `dominant_orientation` | Scalar frame-wide edge orientation (radians); meaningful when `orientedness` is high. |
| `structure_profile` | (3,) frame-level flat/edge/structured fractions; a compact scene signature. |
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

- **`saliency`** — `(G×G)` bottom-up saliency (Itti-Koch style): z-scored channels for
  V-variance (texture), S-mean (colourfulness), center-surround luma contrast, cornerness
  (interest points), and orientation contrast (a cell whose dominant edge orientation
  differs from its surround — the classic orientation pop-out that intensity/colour
  channels miss), averaged and clipped at 0. Center-surround uses a `sal_surround` box.
  Purely per-frame.
- **`text`** — `(G×G)` text likelihood from low-level texture: fine achromatic
  high-frequency contrast in horizontal runs, gated by two orthogonal cues. The first is
  per-cell distribution asymmetry (text is bimodal -- sparse ink on paper -- so its
  `|standardized skew|` is high, while isotropic clutter is symmetric and is suppressed).
  The second is gradient *isotropy*: real text mixes stroke orientations within a cell
  (low structure-tensor coherence, from [`imfeat`](https://github.com/PCJohn/imfeat)),
  so coherent oriented patterns that survive the high-pass -- single edges, parallel rules,
  fences, hatching -- are suppressed (`text_coherence_w`, 0 = off). Tuned for dense/printed
  text (body text, captions, UI); a cue, not OCR. The one intentional text-specific feature
  (see Scope).
- **`motion`** — `(G×G)` motion magnitude vs the previous frame (`None` for a still image).
  See below.
- **structure maps** (`edge_energy`, `coherence`, `cornerness`, `orientation`) — `(G×G)`
  gradient structure-tensor features from [`imfeat`](https://github.com/PCJohn/imfeat),
  computed over the same cells in the same pass as the moments. They read the *gradient
  layout* the intensity moments are blind to: how much edge energy a cell has, whether it is
  anisotropic (one dominant edge) or isotropic clutter, whether it contains a corner, and the
  edge direction. `sharpness` is the global-energy scalar (raw detail/contrast).
- **`focus`** — `(G×G)` edge sharpness = gradient energy per unit intensity variance
  (≈ 1/edge-width²), *contrast-independent*: high where edges are crisp, low where blurred
  or flat. A per-cell focus/defocus and depth-of-field map (unlike `sharpness`, invariant to
  contrast and brightness). Free — reuses the energy and variance already computed.
- **`structure_type`** — `(G×G×3)` soft decomposition `[flat, edge, structured]` summing to 1,
  from the per-cell structure tensor: `presence = energy/(energy+edge_thresh)` (flat vs not),
  then coherence splits a single dominant edge from 2-D structure. `argmax(-1)` gives a hard
  label. Corner and isotropic texture share the eigenvalue signature (the tensor has only two
  eigenvalue DoF — `cornerness == energy·(1−coherence)/2`), so they honestly merge into
  `structured` rather than faking a fourth class.
- **global scene descriptors** — `orientedness` (∈[0,1], global edge anisotropy: high for
  architecture/horizons, ~0 for natural/busy scenes), `dominant_orientation` (radians, the
  frame-wide edge direction, meaningful when `orientedness` is high), and `structure_profile`
  (`(3,)` frame-level flat/edge/structured fractions). Together these give a cheap
  graphic-vs-photographic prior: graphics/documents skew flat+edge with high orientedness,
  natural photos skew structured with low orientedness.

All three are `(G×G)` (`G = 2**grid_exp`, default 32) at thumbnail scale; multiply cell
indices by `shape / G` to map back to source pixels. `saliency` and `text` are
cached on the `FrameStats`, so a duplicate frame reuses them for free; `motion` is recomputed
since it depends on the previous frame.

### Motion map and its local noise floor

`motion = |residual|`, where the residual is the per-cell luma change after fitting and
removing a global gain/bias (`a·prev + b`) — so a uniform brightness shift or auto-exposure
step does **not** read as motion. Raw, the residual still carries per-frame sensor/compression
speckle, so a noise floor is subtracted: `motion = max(|residual| - floor, 0)`, where
`floor = max(motion_floor_k · local_mean|residual|, motion_abs_floor)`. The **local** term
(box average over a `motion_surround` neighborhood, `motion_floor_k` default 1.0) adapts to
regionally-varying noise; the **absolute** term (`motion_abs_floor`, default 1.0 grey level)
guarantees that sub-grey-level change — averaged over a ~60×60-px cell, that is just
sensor/compression dither — reads as no motion on any frame. Set both to 0 for the raw
magnitude. The signed residual is always available as `stats.residual` if you want it before
the absolute value and floor.

After the floor, motion is **structurally validated** (`motion_struct_w`, default 0.7): each
cell is down-weighted by how much its dominant edge *orientation* changed versus the previous
frame. Orientation (the coherence-weighted double-angle vector from imfeat) is
illumination-invariant — brightening a region scales all its gradients equally without
reorienting them — so a regional lighting or shadow shift, which the global affine cannot
absorb, produces luma residual but ~zero orientation change and is suppressed, whereas real
motion moves edges and is kept. In practice this cuts a regional-shadow false response to ~⅓
while leaving genuine motion ~intact. The trade is the aperture case: a uniform region moving
over a uniform background has no orientation change in its interior, so (like the local floor)
this favors motion boundaries; set `motion_struct_w = 0` to disable.

Note the trade the local (center-surround) term makes: it favors motion **boundaries** over
filled interiors, so a large, smoothly-moving region is partly hollowed while its edges remain.
The absolute term is unaffected by this; for a filled-region map set `motion_floor_k = 0` (keep
`motion_abs_floor`) and you get a plain absolute-thresholded magnitude.

### Scope: generic descriptors, not object detectors

framegate deliberately stops at low-level, semantic-free descriptors (the moment grids
and cheap combinations of them: contrast, colorfulness, texture energy, saliency). It
does **not** ship face / barcode / QR / logo detectors.

The one deliberate exception is **`text`**. It composes text-specific priors -- fine
achromatic contrast in horizontal runs, gated by per-cell bimodality (the asymmetric
ink-on-paper distribution that text produces) -- so it is a genuine text *cue*, not a
generic texture descriptor. We make this single exception because text is near-universal
in real imagery and underpins a large class of downstream applications, and because the
cue costs nothing beyond the moments already computed (the bimodality gate reuses the
skew that `imfeat` returns for free). It is still only a cue, not OCR: it is tuned
for dense, achromatic, horizontally-laid-out text (printed body text, captions, dense UI)
and scores large display or colourful text lower.

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
per-frame copy is the float64→float32 cast of the moment grids — if `imfeat`
gains a float32 moment output mode, that copy disappears.

## Video-level optimization (lossless only)

By design, no optimization changes any output:

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

Per-frame latency at 1080p (min over repeats, GC disabled, one frame at a time)
with the **default config** (`thumb=256`, `grid_exp=5`, a 4-level pyramid, `stride=2`).
Absolute numbers scale with CPU clock; the *shape* is consistent across machines.

| Path                          | laptop (charged) | Linux x86 (slow box) |
|-------------------------------|------------------|----------------------|
| `image()` (stateless)         | ~1.4 ms          | ~3.3 ms              |
| `frame()` (temporal)          | ~1.9 ms          | ~4.0 ms              |
| `frame()` + all maps read     | ~2.0 ms          | ~4.2 ms              |
| `frame()` on 50% duplicates   | ~1.4 ms          | ~2.9 ms              |

What moves the number, from `examples/benchmark.py` (figures below are the charged laptop):

- **`thumb` dominates** — it sets the pixel work and scales ~quadratically with it:
  `thumb=64` ≈ 0.14 ms, `128` ≈ 0.57 ms, `192` ≈ 1.19 ms, `256` (default) ≈ 1.44 ms.
  This is the first knob to reach for.
- **Input resolution barely matters** for `image()` (everything downsizes to `thumb`
  first): ~0.75 ms at 360p rising to only ~0.88 ms at 4K. Larger frames cost more only in
  the resize, seen mainly in `frame()` (~0.94 ms → ~1.31 ms at 4K). No need to pre-downscale.
- **`thumb` is the main lever** — cost is quadratic in it, since it sets the pixel count
  actually walked: 64 ≈ 0.21 ms, 128 ≈ 0.36 ms, 192 ≈ 0.56 ms, 256 (default) ≈ 0.83 ms.
- **`stride` is the second lever:** `stride=1` ≈ 1.68 ms, `2` (default) ≈ 0.95 ms,
  `3` ≈ 0.75 ms, `4` ≈ 0.66 ms. It simply subsamples which pixels the single accumulation
  loop visits; the gradient stencil and the cell boundaries stay at full resolution, and
  each cell keeps an identical sample count. What it costs is samples per cell, so the
  binding constraint is `cell_px / stride >= 4` — at the default (`thumb=256`, `grid_exp=5`
  → 8 px cells) that caps you at `stride=2`. Past that the structure maps degrade fast
  (at 4 samples/cell the edge-energy map correlates only ~0.82 with the exact one).
- **`n_levels` (pyramid depth) is nearly free:** `1` ≈ 0.79 ms → `4` ≈ 0.83 ms. Coarse
  levels are exact *sums* of the finer cells' accumulators, so depth costs no extra pass
  and no extra per-pixel work — only the (tiny) reduction over cells. Ask for all of them.
- **`grid_exp` is cheap up to 32×32:** 16×16 ≈ 0.92 ms, 32×32 (default) ≈ 0.94 ms,
  64×64 ≈ 1.45 ms. Finer grids cost more because each cell's accumulators are flushed to
  memory once per row, so the flush traffic grows with the cell count while the pixel work
  stays fixed. `grid_exp=7` (128×128) now *works* — the int16 cell limit that used to
  forbid it was a `tensorstats` artifact — but at `thumb=256` its cells are 2 px wide, so
  it is both expensive and statistically thin. 32×32 remains the sweet spot.

The heavy pixel work (moments, structure tensor, HOG, extrema -- for all three
channels, in one pass) is native; the rest is small-array numpy.
Buffers are preallocated and reused, work is float32, the rolling baseline uses a sort
instead of `np.median`, derived maps are cached, and all array outputs are lazy.

### Tuning for latency

Practical levers, fastest path to a smaller number first:

- **Drop `thumb`.** It is by far the biggest dial. `thumb=128` roughly halves the default
  cost; `thumb=64` is ~10x cheaper. Raise it only when you need richer per-cell statistics
  (more pixels per cell). At `thumb=256`/`grid_exp=5` each finest cell sees 64 px at
  `stride=1`, 16 at `stride=2`.
- **Use `stride` to bound cost at a large `thumb`,** accepting coarser stats — but remember
  it is a gather, so small strides may not pay off for a single grid.
- **Trim `n_levels`** to what your signals actually consume; every extra level is scatter
  work on every pixel.
- **Set `return_frames=False`** if you never read `fs.thumb` / `fs.hsv`. The default
  attaches fresh thumbnail + HSV arrays to every `FrameStats` (two allocations/frame, ~384
  KB at `thumb=256`); turning it off reuses internal scratch and cuts that allocation.
- **Leave `skip_duplicates=True`** (default). Byte-identical consecutive frames reuse the
  previous result behind a cheap strided pre-check — near-free on slideshows, padded streams,
  or held frames.
- **Read signals lazily.** `FrameStats` maps and scalars compute on first access and cache.
  If you only need the cut decision, don't touch `fs.saliency` / `fs.text` /
  `fs.motion` — that is the ~0.15 ms gap between `frame()` and `frame() + all maps`.
- **Disable the garbage collector around your loop.** GC pauses are the dominant source of
  latency *spikes* (not the steady cost): `gc.disable()` before the loop with a periodic
  `gc.collect()` flattens the tail. `examples/visualize.py` shows the pattern.
- **Consider `cv2.setNumThreads(1)`.** These are small-array ops, so OpenCV's default
  multithreading can add scheduling variance (most visible in the micro-sweeps on many-core
  machines). Pinning to one thread usually steadies — and sometimes slightly speeds — them.
  The library never sets this globally, since a library shouldn't mutate process-wide state;
  the benchmark interleaves its sweeps so clock drift is spread evenly across configs.

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
│   ├── gate.py            # Gate facade + lossless duplicate skip
│   ├── publish.py         # Publisher + Packet (pub/sub node; drops blank/frozen frames)
│   ├── shotmem.py         # ShotMemory + ShotTracker: shot re-identification (L2)
│   ├── reid.py            # per-bit Bernoulli shot model (ShotProfile, ShotScorer)
│   └── longterm.py        # L3 mmap prototype-index stub
├── examples/
│   ├── visualize.py       # live matplotlib dashboard (frame + maps + signals + shot/group)
│   ├── shots.py           # live shot re-ID viewer: groups build up as the video plays
│   ├── benchmark.py       # latency measurement
│   └── publish.py         # Publisher demo: drops frames, prints shot_id + shot_group_id
└── tests/
    ├── synth.py           # synthetic scene + pattern builders
    ├── test_accuracy.py   # feature functionality: cuts, blank, freeze, fade, flicker, dedup
    ├── test_robustness.py # degenerate inputs, value ranges, invariances, config sweeps
    ├── test_config.py     # YAML template parity, overrides, immutability
    ├── test_latency.py    # per-frame latency budget (min-of-repeats, GC off)
    ├── test_publish.py    # Publisher drop policy + frame_id / shot_id / shot_group_id
    ├── test_structure.py  # structure-tensor feature plumbing
    ├── test_reid.py       # per-bit Bernoulli scorer: exactness, loose-match, latency
    └── test_shotmem.py    # ShotMemory + ShotTracker: group ids, recall+score, metadata, ABAB
```

All visualization and timing code lives in `examples/` — the library itself contains
no matplotlib or benchmarking code.

## Examples & tests

```bash
python examples/visualize.py path/to/video.mp4     # needs [viz]
python examples/benchmark.py                        # synthetic, or pass a video path
python examples/publish.py                          # publishing gate (synthetic, or pass a video)
python examples/shots.py                            # live shot re-ID viewer (synthetic, or pass a video)
pytest                                              # needs [dev]; -s prints latencies
```

The examples and tests import `framegate` exactly as a downstream user would.