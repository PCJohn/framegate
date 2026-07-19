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

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .config import GateConfig
from .gate import Gate
from .shotmem import ShotMemory, ShotRef
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

    def __init__(self, cfg: Optional[GateConfig] = None):
        self.cfg = cfg or GateConfig()
        self._gate = Gate(self.cfg)
        self._subs: List[Callable[[Packet], None]] = []
        self._frame_id = -1
        self._shot_id = 0
        self._mem = ShotMemory(self.cfg.reid_z)
        self._group = -1  # current shot's group id (-1 until the first shot opens)
        # cut is confirmed one frame late, so at a cut `_last` is the NEW shot's first
        # frame and `_prev` is the OLD shot's last frame.
        self._last: Optional[ShotRef] = None  # previous published frame (t-1)
        self._prev: Optional[ShotRef] = None  # published frame before that (t-2)
        # re-ID kernel: reuse the cut detector's NCC + MAD-z between two descriptors
        self._score = lambda q, r: self._gate.shot_z(q.luma, q.color, r.luma, r.color)

    def subscribe(self, fn: Callable[[Packet], None]) -> None:
        """Register a callback invoked with each published Packet."""
        self._subs.append(fn)

    def _emit(self, pkt: Packet) -> None:
        for fn in self._subs:
            fn(pkt)

    def publish(self, frame: np.ndarray) -> Optional[Packet]:
        """Process the next frame; return a Packet if it passes the gate, else None.
        Dropped: blank frames and freezes (which include byte/near duplicates)."""
        self._frame_id += 1
        stats, signals = self._gate.frame(frame)
        if stats.blank or signals.freeze:
            return None
        desc = ShotRef(stats.v_cell_mean, stats.color_mean)
        if self._group < 0:  # first shot opens as group 0
            self._group, _ = self._mem.match(desc, self._score)
        elif (
            signals.cut
        ):  # cut (confirmed 1 frame late) closes this shot, opens the next
            self._shot_id += 1
            if self._prev is not None:
                self._mem.refresh(
                    self._group, self._prev
                )  # old shot's last clean frame
            first = (
                self._last if self._last is not None else desc
            )  # new shot's 1st frame
            self._group, _ = self._mem.match(first, self._score)
        self._prev, self._last = self._last, desc
        pkt = Packet(self._frame_id, self._shot_id, self._group, frame, stats, signals)
        self._emit(pkt)
        return pkt
