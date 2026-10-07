# How the gate got fast

An engineering log of the latency work on `framegate`: where it started, the shape of a
frame now, what each step bought, what was tried and not kept, the lessons, how it is
measured, and the settings to run it with. imfeat's own log
([imfeat/docs/OPTIMIZATION.md](https://github.com/PCJohn/imfeat/blob/main/docs/OPTIMIZATION.md))
covers the pass itself and fastdet's ([fastdet/docs/OPTIMIZATION.md](https://github.com/PCJohn/fastdet/blob/main/docs/OPTIMIZATION.md))
the model stage; this one covers the gate around them, and refers to those for their
numbers.

The constraints throughout: no optimisation changes an output (the one exception,
`fast_static`, is documented in the README and can be switched off); nothing video-level
moves into imfeat, which stays a single-image library; framegate itself stays pure Python
over numpy, OpenCV and the two native libraries — no C++ build of its own; nothing tuned
to one machine. Every number below was measured, and says where: the laptop is the
maintainer's 22-thread Windows machine (AVX2); the development VM a shared two-core Linux
machine whose timings wander by ±5–10%.

## Starting point

The first version (June 2026) computed per-cell colour moments with `tensorstats` and,
later, structure-tensor features with `structstats`: two passes over a 256-px thumbnail that
OpenCV had resized and converted, a 32×32 grid, and a temporal layer in numpy over the cell
means. What was there from the start and is still the design: a stateless per-frame layer
producing lazy maps, a stateless-per-call temporal layer over a rolling window, buffers
preallocated and reused, float32 throughout, the robust baseline sorted rather than
`np.median`'d, and a benchmark that interleaves configurations so clock drift is spread
evenly. The READMEs of June to August recorded a frame at `thumb=256` costing about 0.8 ms,
with the pyramid depth nearly free (`n_levels` 1 → 4: 0.79 → 0.83 ms) and the grid cheap up
to 32×32 (16×16 0.92, 32×32 0.94, 64×64 1.45 ms).

What it did not yet have: one pass for everything, threads, a thumbnail that follows the
frame, the resize and the colour conversion inside the pass, a learned map, or a temporal
layer that reuses what it already computed.

## Terms

| term | meaning |
|---|---|
| pass | imfeat's single traversal of the thumbnail: every per-cell feature, the summary, the hashes |
| thumbnail | what the pass runs on: `thumb` px square, or a size the policy derives from the frame |
| grid, level | the 64×64 finest cell grid (`grid_exp=6`) and the dyadic levels under it (`n_levels=6`) |
| stride | the sampling stride on the thumbnail; 1 = every pixel |
| FrameStats | the per-frame object: the grids as float32, the lazy maps, the model maps |
| temporal layer | `StreamAnalyzer`: cut, freeze and fade from a window of FrameStats |
| model stage | the fastdet scorer on the pass's maps, inside `FrameGate.process()` |
| gate frame | `gate.frame(frame)` end to end: duplicate check, pass, model stage, stats, temporal layer |
| operating point | 720p at `"pow2-fit"` (512×320 thumbnail), HSV, stride 1, `feat_threads=2`, the text model on 2 threads |

## Result

A gate frame on the laptop at the operating point (`examples/gate_loop.py`, medians over
100–600 frames, GC disabled): **2.14 ms** (p90 2.38; the minimum of 0.25 is a held frame
the duplicate check settles), of which the model stage is 0.22 ms and the imfeat pass about
1.8 ms; on an earlier clip and model 2.9 ms with a 0.37 ms model stage. On the mainline
as pushed on 7 October, the earlier clip with the tuning sweep's model `de716422c0b5`,
five runs back to back: 2.47–2.75 ms (p90 2.9–4.3), the model stage 0.37 ms on 2 scorer
threads (0.35–0.41 on 8), the lazy maps the dashboard reads 0.31 ms more. Through `examples/visualize.py`, which draws between frames,
the gate median reads 4.9 ms and 5.4 with every map read, the model stage 0.6 ms (2037
frames; 4.83 / 5.36 / 0.58 on 561 frames of an earlier run).

For scale, the same pass on the laptop as recorded in the README: a 1080p frame to HSV
features at 1024×576 costs 7.5 / 4.0 / 2.4 ms on 1 / 2 / 4 threads (10.8 / 5.7 / 3.2 at the
1024 square), the resize inside the pass 0.5 ms on two threads against 2–3 ms for
`cv2.resize` on OpenCV's 22 threads.

Where a gate frame goes at the operating point, two threads: the pass ~85%, the model stage
~10%, the rest — the duplicate check, the float32 conversions of the grids, the model maps'
handover, `FrameStats`, the temporal layer — a few percent, each item tens of microseconds.

## Architecture

A frame, in order (`Gate.frame` → `FrameGate.process` → `StreamAnalyzer.update`):

1. **Duplicate check.** `np.array_equal` against the previous frame, decided in a few
   microseconds for a frame that differs (probe rows), paid in full only for a duplicate,
   which then reuses the previous stats entirely.
2. **The pass.** The BGR frame goes into imfeat whole; the pass makes the thumbnail
   (`cv2.resize(INTER_AREA)`'s bytes) and converts it to HSV (`cv2.cvtColor`'s bytes) as it
   reads the rows, on `feat_threads` bands, and returns the pyramid of maps, the per-level
   summaries and the hashes — views of a pooled block, no copies.
3. **The model stage.** Any `.fdt` model scores the pass's maps in place
   (`Detector.score_raw`), on its own small pool, first — it walks megabytes of maps and
   would evict the grids the next steps read.
4. **The grids.** The moment grids are cast to float32 (the one unavoidable copy), the
   cell-mean planes made contiguous once, the blank test read from the summary (two values,
   no reduction), and `FrameStats` is built; every map on it is lazy and cached.
5. **The temporal layer.** Cut, freeze and fade from the cell-mean luma against a rolling
   window, with the previous frame's statistics carried rather than recomputed and the
   motion-compensation search skipped when the zero-shift correlation already says the
   frame is static.

## What worked

Chronologically. Numbers are the laptop's unless marked VM; the early ones are those the
README recorded at the time.

| step | change | measured effect |
|---|---|---|
| 1 | June 2026: buffers preallocated, `ravel()`s removed from the per-frame path, derived maps cached, outputs lazy, the rolling baseline by an explicit sort | the ~0.8 ms frame at `thumb=256`; the lazy maps the ~0.15 ms gap between `frame()` and `frame()` plus every map |
| 2 | One unified imfeat pass replaces `tensorstats` + `structstats` (July) | one traversal of the thumbnail for the moments and the structure tensor; the separate FAST corner detector of the June version gave way to imfeat's cornerness map |
| 3 | imfeat's threads (`feat_threads`), output bit-identical at any count; the z-score computation parallelised | the pass became the parallel lever; the gate itself stayed single-threaded |
| 4 | Blankness read from imfeat's per-level summary instead of a numpy reduction over the grid | O(1) in the grid size |
| 5 | The operating point moved to `thumb=1024`, 64×64 grid, six levels, stride 4, then stride 1 when the pass became shareable with fastdet | one finest cell on ~30×17 source pixels at 1080p; stride 1 is what the models are trained on |
| 6 | fastdet models run on the pass the gate already makes (`models/*.fdt`; a `text.fdt` replaces the heuristic text map) | a learned map for the cost of its scorer alone: 0.37 ms per frame on the laptop with the first model and clip, 0.22 with the current ones (fastdet's log has the steps between) |
| 7 | BGR → HSV inside imfeat's pass, then the `INTER_AREA` thumbnail inside it, then the thumbnail sized from the frame (`"pow2"`, then `"pow2-fit"`) | the two OpenCV passes over the frame are gone; a 720p frame runs at 512×320, a 1080p one at 1024×576 — the pass about 30% cheaper than at the 1024 square (7.5 against 10.8 ms on one thread) |
| 8 | The duplicate detector: probe rows, a hot row, a block scan (October) | the old check compared a 2-D stride of the frame whose innermost run is one pixel's three channels, tens of microseconds on every frame; a differing frame is now rejected in microseconds, and `test_dupgate.py` pins the predicate to `np.array_equal` |
| 9 | The stream update: the previous frame's luma statistics carried, ring buffers instead of deques, the shift search's working arrays kept (a view of column bands so numpy sums each window in the order it sums a copy), the colour vector's trigonometry in one scratch, `ori_change` lazy | the same bits (`test_stream_exact.py`); `update()` per frame −33% to −47% on four clips on the VM, −52% on a film clip on the laptop |
| 10 | The flicker signal removed | one `rfft` of the brightness history per frame, for a signal nothing consumed |
| 11 | The temporal layer measured live rather than warm: the cell-mean planes made contiguous once in `process()`, `ori_change` computed only when `motion` reads it, the trigonometry and products into kept scratch, fewer numpy calls; the model stage moved before the grid work | VM, live per frame (the lazy inputs plus `update()`): frames without the motion search 243 / 234 / 218 → 121 / 124 / 134 µs on three clips, frames with it 624 / 725 / 588 → 473 / 514 / 448 µs; the grids are still in cache when the temporal layer reads them |
| 12 | A separate thread count for the scorer (`model_threads`) | the text model (a 0.2 ms pass): 2 threads 0.23 ms, 8 threads 0.31, 16 threads 0.40 for the scorer's pass (the laptop, 5 October; recorded in the *model_threads* commit's message) — a control, not a lever. The tuning sweep's model `de716422c0b5` (a 0.33 ms pass) on the pushed mainline, 7 October, five runs back to back with the gate frame steady at 2.47–2.75 ms: 2 threads 0.37 / 0.37 / 0.37 ms, 8 threads 0.41 / 0.35, 16 threads 0.47 in an earlier run — two and eight within noise, sixteen worse, the same shape |

**Step 7, the front end in imfeat.** The gate used to resize with `cv2.resize` and convert
with `cv2.cvtColor` before the pass: two full passes over the frame, each writing an image
the next step read back. Both now happen inside imfeat's row gather, per band, with exactly
OpenCV's bytes — checked on every colour and on frames of many sizes (imfeat's tests) and,
here, by `test_accuracy.py` comparing the fused path against the OpenCV path on the gate's
own outputs. The thumbnail policy then let the size follow the frame: `"pow2-fit"` keeps the
frame's shape inside the square of its shorter side's power of two, so a 720p frame costs a
third of a 1080p one instead of being upscaled to 1024, and a 1080p frame runs on 44% fewer
pixels than the square. Grayscale frames, frames smaller than the thumbnail and the
`"nearest"` filter still go through OpenCV, with the same numbers out either way.

**Step 8, the duplicate detector.** `skip_duplicates` reuses the previous stats for a
byte-identical frame, so every frame pays a comparison. A frame that differs is now rejected
by the first probed row that differs — the row that rejected the last differing frame, then
four fixed rows — each compared as one contiguous run of bytes so the comparison stops at
the first byte; a frame that differs in none of them is scanned in blocks of rows, and the
first differing row of the differing block becomes the next frame's probe, so a change
confined to one band costs one scan, then one probe per frame. Only a duplicate pays the
whole comparison. The predicate is exactly `np.array_equal`.

**Step 9, the stream update.** Each pairwise step of the temporal layer (the affine fit, the
zero-shift correlation, the two standard deviations) is written on statistics taken once per
frame (`_Luma`: the mean, the centred map, its sum of squares) and carried over when the
frame becomes the previous one — the original's formula operation for operation, with the
reductions it would recompute replaced by the ones already taken on the same arrays, so the
bits are the same (`test_stream_exact.py` compares every signal against the original
implementation frame by frame). The rolling windows are rings in one float32 array; the
shift search keeps its working arrays; the chroma vector's `cos` and `sin` go into one
scratch and are summed in one call.

## What did not work, or was not kept

* **Asking imfeat for planar maps** (September 2026; not in the tree). The gate's
  computers were to take imfeat's planar layout so that a hosted model could score the
  planes where they lay (a 3.7 MB packed copy per frame saved: 0.58 ms on the laptop). The
  layout was dropped on imfeat's side (its log); the saving came anyway when fastdet's scorer
  learned to read the cell-major maps in place, with no change to the gate beyond the call.
* **A native shift-search kernel** (`framegate-6`, declined). The motion-compensation
  search (`best_shift`, a normalised cross-correlation over ±3 cells) is five passes of
  numpy over 49 windows that numpy cannot fuse, about 300 µs per searched frame on the VM
  and most of `update()`'s time on footage that moves. A Highway kernel of the same
  expression — every sum in numpy's own pairwise order, eight accumulators as eight lanes,
  so its bits were numpy's on every target — took it to about 40 µs and `update()` on
  moving footage from ~360 to ~150 µs. It was not adopted: it would have given framegate a
  C++ build of its own (scikit-build-core, nanobind, Highway), and the maintainer's rule is
  that the native code lives in imfeat and fastdet. The numpy search with kept scratch
  arrays is what shipped (step 9); `fast_static` skips it on near-static frames.
* **Moving video-level work into imfeat.** The same kernel could have lived in imfeat's
  extension; it was not moved there either, by the rule that imfeat stays a single-image
  library.
* **The resize by periods** (imfeat side; built, measured, reverted by the maintainer):
  imfeat's log has the numbers. At the gate's operating point the resize inside a two-thread
  pass measured 0.32 ms on the laptop with that kernel, so the saving over the window kernel
  that stayed was a few tenths of a millisecond at most.
* **Lazy conversion of the coarse grid levels.** The float64 → float32 cast of every level's
  moments costs about 20 µs on the VM; making the coarse levels lazy was measured and not
  done, for the complexity of the laziness against the size of the gain. The casts are set by
  imfeat's output layout (float64 moments at full precision for the oracle tests).
* **Saliency in C++.** The Itti–Koch-style saliency map is a grid-level computation (439 µs
  on the VM when read) and could be an exact C++ port, but it is read lazily and only by
  consumers that want it; where such a port would live is undecided.
* **Near-duplicate skipping by signature** is deliberately not in the tree: it is not
  lossless.

## Lessons

1. **The pass is the frame.** At the operating point imfeat is ~85% of a gate frame and the
   model 10%; everything else in the gate is tens of microseconds. Work on the gate's own
   Python pays only once a frame is a few milliseconds.
2. **Fuse the passes over the frame, not the operations on the grid.** Resize and colour
   conversion inside the pass removed two full-frame passes and their intermediate images;
   that was worth more than any grid-level change.
3. **Size the thumbnail from the frame.** The pixel work is quadratic in the thumbnail's
   side; a policy that keeps a 720p frame at 512 px instead of upscaling it to 1024 is the
   cheapest change in the whole log.
4. **Carry statistics, do not recompute them.** The temporal layer's pairwise formulas can
   be written on sums taken once per frame, with the same bits; the proof is a test that
   compares every signal against the original implementation.
5. **Measure live, not warm.** Warm in isolation, the temporal layer read as 85 µs per
   frame; inside `gate.frame()` after the pass it was 240 µs, because numpy's dispatch runs
   cold after a C++ pass and every call costs several times its warm price. The layer made
   about 45 calls; halving the count was what halved the time.
6. **Order the steps for the cache.** The model stage walks megabytes; putting it before the
   grid work keeps the grids warm for the temporal layer.
7. **Prove losslessness by construction.** `np.array_equal` as the duplicate predicate, the
   original formulas for the stream, the OpenCV bytes for the front end — each optimisation
   comes with a test that pins it to what it replaced.
8. **A thread count is a control.** The scorer is fastest, or as fast, on two threads on
   the laptop for both models measured (0.2 and 0.33 ms passes); a wider split costs more
   in synchronisation than it saves, and sixteen threads lose for both. imfeat's threads
   are the lever, up to four.

## Measuring

* **`python examples/gate_loop.py clip.mp4 --model text.fdt --threads 2 --model-threads 2
  --frames N`**: `gate.frame()` per frame, the model stage and `score_maps` timed by
  monkey-patching, medians and p90, GC disabled; `--pyspy out.json` attaches py-spy once the
  gate is warm. The number of record.
* **`python examples/gate_stages.py clip.mp4 --threads 2 --model text.fdt`**: where a gate frame
  goes stage by stage — the pass (frame in, thumbnail made inside), the model stage, the rest
  of `process()`, the temporal layer — and the pass on a ready thumbnail alongside, which
  isolates the in-pass resize. Frames the gate skips as duplicates are left out of the
  medians and counted.
* **`examples/benchmark.py [video]`**: the sweeps over sizes, grids and strides, where a
  frame goes internally, what each lazy map costs, and how imfeat's pool and OpenCV's
  interact; minimum over repeats, configurations interleaved.
* **`examples/visualize.py --model text.fdt --model-threads N`**: the demo, with the gate
  median, the median with every map read, and the model stage's median; it draws between
  frames, so its numbers run higher than the loop's.
* **`pytest -s tests/test_latency.py`**: budgets generous enough for CI, with the numbers
  printed.

## Verifying

* The suite (203 tests): the fused front end against the OpenCV path on the gate's outputs
  (`test_accuracy.py`); the duplicate detector against `np.array_equal` over frames that
  differ in one row, one byte, one block, in shape or dtype (`test_dupgate.py`); every
  temporal signal against the original implementation frame by frame
  (`test_stream_exact.py`); the model maps against fastdet's own `predict_proba` bit for
  bit, at every thread count (`test_models.py`); the thumbnail policies; the structure maps;
  robustness and accuracy on synthetic streams.
* ruff (E, F, W, I, UP, B, SIM, RUF), black, mypy on `src` and `tests`.

## Recommended configuration

* The defaults are the measured operating point: `thumb="pow2-fit"`, `resize_interp="area"`,
  `stride=1`, `grid_exp=6`, `n_levels=6`, `feat_threads=2`, `model_threads=0` (= 2),
  `skip_duplicates=True`, `fast_static=True`. They are what the fastdet models are trained
  on; a model's front end is checked against the config at load.
* **Cheaper, without a model:** `thumb` first (`thumb=512` on 1080p is about a third of the
  square's cost; `thumb=256` a fraction of that), then `stride` (the pixel work scales with
  `1/stride²`, with `cell_px / stride >= 4` as the floor), then `n_levels` to what the signals
  consume. `GateConfig(stride=4, resize_interp="nearest")` keeps the grid at a sixteenth of
  the samples; `GateConfig(thumb=256, stride=2, grid_exp=5, n_levels=4)` is the old 256 px
  point.
* **Threads:** `feat_threads=2` when the machine is shared, 4 when it is not; leave
  `model_threads` alone (= 2): two threads were best or equal-best for both models measured
  and sixteen worse for both; `examples/gate_loop.py --model-threads N` checks another
  model or machine.
* **`return_frames=False`** if `fs.thumb` / `fs.hsv` are never read: it is the one
  allocation per frame (3 MB at 1024 px).
* **Read signals lazily**: a consumer that needs only the cut decision should not touch the
  maps; each is computed on first access and cached.
* **Disable the garbage collector around the loop** with a periodic `gc.collect()`; GC
  pauses are the latency spikes, not the steady cost. `examples/visualize.py` shows the
  pattern.
* **`cv2.setNumThreads(1)`** in the host steadies the small-array OpenCV calls on many-core
  machines; the library never sets it.
* **Call `Gate.close()`** (or `Publisher.close()`) when a stream ends, and before interpreter
  shutdown on Windows: it joins imfeat's and the scorer's pools.

## Open ideas

* **imfeat's pass**, since it is 85% of the frame: the exact items in imfeat's log (batched
  folds with the roll-up fused in; the serial tail at two threads).
* **Warm maps for the model stage**: the 0.1 ms the scorer pays reading cold maps imfeat's
  threads wrote (fastdet's log); it crosses the library boundary either way.
* **The grid conversions**: a float32 moment output from imfeat would remove the per-frame
  casts (tens of microseconds); the casts exist because imfeat keeps the moments at full
  precision for its oracle tests.
* **Saliency as an exact native kernel**, if a consumer reads it every frame.
