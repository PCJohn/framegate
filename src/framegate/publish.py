"""Publishing layer: turns the gate into a pub/sub-style node.

A Publisher consumes frames in order, runs the Gate, and emits a Packet only for
frames worth downstream work -- dropping blank frames and freezes (a byte- or
near-identical duplicate surfaces as `freeze`, so it is dropped too). A subscriber
can therefore assume every Packet it receives is worth the heavy pipeline.

Each Packet carries the frame, its FrameStats (the feature pyramid + all per-frame
maps), the TemporalSignals, and inferred stream metadata: frame_id, shot_id, and
shot_group_id. shot_id counts shots; shot_group_id is shared by every recurrence of
the same shot (see shotmem.py), so a dialogue that cross-cuts between the same three
setups emits shot_ids 0,1,2,3,4,5,... with shot_group_ids 0,1,2,0,1,2,... The emit
path is a plain callback fan-out, so a real transport (zmq, asyncio queue, ROS, ...)
can replace `_emit` later without touching the gate or the drop policy.

    pub = Publisher()
    pub.subscribe(handle)                 # optional push interface
    for frame in source:
        pkt = pub.publish(frame)          # pull interface; None if dropped
        if pkt is not None:
            run_detector(pkt.frame, saliency=pkt.stats.saliency)
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import GateConfig
from .gate import Gate
from .shotmem import Shot, ShotTracker
from .stats import FrameStats
from .stream import TemporalSignals


@dataclass
class Packet:
    """What the gate publishes for a frame worth processing. `stats` carries the
    feature pyramid and every per-frame map; `signals` the temporal scalars."""

    frame_id: int  # index over ALL input frames incl. dropped, so gaps stay visible
    shot_id: int  # increments on each detected cut (first frame of the new shot)
    shot_group_id: int  # shared by recurrences of the same shot (re-identification)
    frame: np.ndarray
    stats: FrameStats
    signals: TemporalSignals


class Publisher:
    """Stateful gate node: `publish(frame)` in, `Packet` (or `None`) out. Blank and
    frozen/duplicate frames are dropped, so downstream only ever sees frames worth
    processing. Not thread-safe -- use one Publisher per stream."""

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self._gate = Gate(self.cfg)
        self._subs: list[Callable[[Packet], None]] = []
        self._frame_id = -1
        self._shots = ShotTracker(self.cfg)

    def subscribe(self, fn: Callable[[Packet], None]) -> None:
        """Register a callback invoked with each published Packet."""
        self._subs.append(fn)

    def _emit(self, pkt: Packet) -> None:
        for fn in self._subs:
            fn(pkt)

    def publish(self, frame: np.ndarray) -> Packet | None:
        """Process the next frame; return a Packet if it passes the gate, else None.
        Dropped: blank frames and freezes (which include byte/near duplicates)."""
        self._frame_id += 1
        stats, signals = self._gate.frame(frame)
        if stats.blank or signals.freeze:
            return None
        shot_id, group_id = self._shots.update(stats, signals, self._frame_id)
        pkt = Packet(self._frame_id, shot_id, group_id, frame, stats, signals)
        self._emit(pkt)
        return pkt

    def close(self) -> None:
        """End of stream: close the final shot so it appears in `shots`. Idempotent."""
        self._shots.close()

    @property
    def shots(self) -> list[Shot]:
        """Closed shots so far. The one in progress appears after `close()`."""
        return self._shots.shots
