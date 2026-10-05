"""High-level facade. Most users only need this.

    gate = Gate()                       # or Gate(GateConfig.from_yaml("my.yaml"))
    stats = gate.image(img)             # single image -> FrameStats
    for frame in video:
        stats, signals = gate.frame(frame)   # video -> FrameStats + TemporalSignals

`image()` is stateless. `frame()` runs the temporal layer and adds one lossless
optimization: byte-identical consecutive frames reuse the previous result instead
of recomputing (the stats of an identical frame are identical, so this changes no
output). It is gated by a cheap strided pre-check, so distinct frames pay only a
few microseconds.

Caveat: duplicate detection holds a reference to the previous frame. Sources that
decode into one reused buffer in place (rare; OpenCV/imageio/decord all return
fresh arrays) would defeat it -- pass copies or set skip_duplicates=False then.
"""

import numpy as np

from .config import GateConfig
from .stats import FrameGate, FrameStats
from .stream import StreamAnalyzer


class DuplicateDetector:
    """The front gate: is this frame byte-identical to the last one? Decided in a few
    microseconds when it is not, which is nearly every frame of real video.

    The predicate is exactly ``np.array_equal(a, b)``. A frame that differs is rejected by
    the first probed row that differs: the row that rejected the last differing frame, then
    four fixed rows (halfway, the quarters, the last), each one contiguous run compared as
    bytes, so the comparison stops at the first byte that differs. A frame that differs in
    none of them is scanned in blocks of rows until the block that differs, whose first
    differing row becomes the probe for the next frame: a change that sits in one band costs
    one scan, then one probe per frame. Only a duplicate pays the whole comparison. The rows
    and blocks are fixed when a frame shape is first seen and re-derived only when it changes.
    """

    __slots__ = ("_blocks", "_dtype", "_hot", "_probes", "_rows", "_shape")

    def __init__(self) -> None:
        self._shape: tuple = ()
        self._dtype: np.dtype | None = None
        self._rows: tuple = (0, 0, 0, 0)
        self._hot = 0  # the probe row that rejected the last differing frame
        # rows can be probed: a first axis to index, and plain bytes
        self._probes = False
        self._blocks: tuple = ()  # (first row, end row) of the blocks the scan walks

    def _fit(self, a: np.ndarray) -> None:
        """The probe rows and scan blocks for this frame shape. Rows are probed as bytes, which
        stands in for ``==`` on the usual integer frames; an object dtype compares pointers,
        so it goes straight to the full comparison (floats are safe: a byte difference such
        as +0.0 against -0.0 only ever rejects, and a rejected frame is simply processed).
        """
        self._shape, self._dtype = a.shape, a.dtype
        n = a.shape[0] if a.ndim else 0
        self._rows = (n // 2, n // 4, (3 * n) // 4, n - 1) if n else (0, 0, 0, 0)
        self._hot = self._rows[0]
        self._probes = n > 0 and not a.dtype.hasobject
        nb = min(16, n)
        self._blocks = tuple((n * i // nb, n * (i + 1) // nb) for i in range(nb))

    def _scan(self, a: np.ndarray, b: np.ndarray) -> bool:
        """The cold path, after every probe matched: compare block by block, stopping at the
        first block that differs; its first differing row is the next frame's first probe.
        """
        for r0, r1 in self._blocks:
            if not np.array_equal(a[r0:r1], b[r0:r1]):
                for r in range(r0, r1):
                    if a[r].tobytes() != b[r].tobytes():
                        self._hot = r
                        break
                return False
        return True

    def same(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether ``a`` and ``b`` hold the same values (``np.array_equal``), for the cost of a
        row or two when they do not. Two names for one buffer are never "the same frame":
        the previous frame's bytes are gone, so there is nothing to reuse."""
        if a is b:
            return False
        shape = a.shape
        if shape != b.shape or a.dtype != b.dtype:
            return False
        if shape != self._shape or a.dtype != self._dtype:
            self._fit(a)
        if not self._probes:
            return bool(np.array_equal(a, b))
        hot = self._hot
        if a[hot].tobytes() != b[hot].tobytes():
            return False
        r0, r1, r2, r3 = self._rows
        if r0 != hot and a[r0].tobytes() != b[r0].tobytes():
            self._hot = r0
            return False
        if r1 != hot and a[r1].tobytes() != b[r1].tobytes():
            self._hot = r1
            return False
        if r2 != hot and a[r2].tobytes() != b[r2].tobytes():
            self._hot = r2
            return False
        if r3 != hot and a[r3].tobytes() != b[r3].tobytes():
            self._hot = r3
            return False
        return self._scan(a, b)


class Gate:
    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self._gate = FrameGate(self.cfg)
        self._stream = StreamAnalyzer(self.cfg)
        self._last_frame: np.ndarray | None = None
        self._last_stats: FrameStats | None = None
        self._dup = DuplicateDetector()

    def image(self, img: np.ndarray) -> FrameStats:
        """Analyze a single image. No temporal state is touched."""
        return self._gate.process(img)

    @property
    def models(self) -> list[str]:
        """Names of the fastdet models loaded (see models.py); their maps are on
        `FrameStats.model_maps`."""
        return self._gate.models

    @property
    def model_threads(self) -> dict[str, int]:
        """Threads each model's scorer runs on (`model_threads`, or `feat_threads`)."""
        return self._gate.model_threads

    def close(self) -> None:
        """Release the imfeat pool and the models (their threads are joined). Call it
        when a stream ends in a long-running host, and before exit on Windows."""
        self._gate.close()

    def frame(self, frame: np.ndarray) -> tuple:
        """Analyze the next video frame. Returns (FrameStats, TemporalSignals)."""
        if (
            self.cfg.skip_duplicates
            and self._last_frame is not None
            and self._dup.same(frame, self._last_frame)
        ):
            fs = self._last_stats
        else:
            fs = self._gate.process(frame)
            self._last_frame, self._last_stats = frame, fs
        return fs, self._stream.update(fs)
