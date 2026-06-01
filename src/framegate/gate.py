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


def _identical(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if not np.array_equal(a[::16, ::16], b[::16, ::16]):   # cheap reject for distinct frames
        return False
    return np.array_equal(a, b)


class Gate:
    def __init__(self, cfg: GateConfig = None):
        self.cfg = cfg or GateConfig()
        self._gate = FrameGate(self.cfg)
        self._stream = StreamAnalyzer(self.cfg)
        self._last_frame = None
        self._last_stats = None

    def image(self, img: np.ndarray) -> FrameStats:
        """Analyze a single image. No temporal state is touched."""
        return self._gate.process(img)

    def frame(self, frame: np.ndarray):
        """Analyze the next video frame. Returns (FrameStats, TemporalSignals)."""
        if (self.cfg.skip_duplicates and self._last_frame is not None
                and _identical(frame, self._last_frame)):
            fs = self._last_stats
        else:
            fs = self._gate.process(frame)
            self._last_frame, self._last_stats = frame, fs
        return fs, self._stream.update(fs)
