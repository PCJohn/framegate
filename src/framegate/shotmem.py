"""Shot re-identification memory: recognise when a shot re-appears (the wide/A/B
shots a dialogue or cross-cut keeps returning to), so each shot gets both a unique
`shot_id` and a `shot_group_id` shared by every recurrence of the same shot.

A shot is represented by one reference descriptor -- the cell-mean luma layout +
global colour of its most recent frame (the exact descriptors the cut detector
already computes). At each cut the new shot's first frame is compared against the
stored references with the *same* NCC + median/MAD kernel as cut detection (see
StreamAnalyzer.shot_z), reusing the existing machinery with a looser threshold.

Retrieval is deliberately isolated behind `_Index`. The default `LinearIndex` is an
O(K) scan; swapping in an LSH/ANN index for O(1) lookup is a surgical replacement --
it only has to preserve the add()/query() interface, and ShotMemory and the match
kernel stay untouched.
"""

from collections import namedtuple
from typing import Callable, Dict, Iterable, Optional, Tuple

ShotRef = namedtuple("ShotRef", "luma color")  # (G,G) cell-mean luma, (3,) colour


def _copy(d: ShotRef) -> ShotRef:
    return ShotRef(d.luma.copy(), d.color.copy())


class LinearIndex:
    """O(K) retrieval: every group is a candidate. The swap point for constant-time
    retrieval -- a replacement (e.g. LSH over the luma layout) only needs the same
    add()/query() interface; the descriptor argument is ignored here."""

    def __init__(self) -> None:
        self._ids: list = []

    def add(self, group_id: int, desc: ShotRef) -> None:
        self._ids.append(group_id)

    def query(self, desc: ShotRef) -> Iterable[int]:
        return self._ids


class ShotMemory:
    """Assigns shot group ids by nearest stored reference under `threshold`. `match`
    is called once per shot (at its opening cut) with the shot's first-frame
    descriptor and the re-ID score fn; `refresh` updates a group's reference to its
    latest frame so a slowly drifting shot keeps matching."""

    def __init__(self, threshold: float, index: Optional[LinearIndex] = None) -> None:
        self._thr = threshold
        self._index = index if index is not None else LinearIndex()
        self._refs: Dict[int, ShotRef] = {}
        self._n = 0

    def match(
        self, desc: ShotRef, score: Callable[[ShotRef, ShotRef], float]
    ) -> Tuple[int, bool]:
        """Return (group_id, is_new). Nearest stored reference wins if its score is
        <= threshold; otherwise a new group is created with `desc` as its reference."""
        best_id, best = None, float("inf")
        for gid in self._index.query(desc):
            s = score(desc, self._refs[gid])
            if s < best:
                best, best_id = s, gid
        if best_id is not None and best <= self._thr:
            return best_id, False
        gid = self._n
        self._n += 1
        self._refs[gid] = _copy(desc)
        self._index.add(gid, desc)
        return gid, True

    def refresh(self, group_id: int, desc: ShotRef) -> None:
        self._refs[group_id] = _copy(desc)

    def __len__(self) -> int:
        return self._n


class ShotTracker:
    """Turns a stream of per-frame gate outputs into (shot_id, shot_group_id): shot_id
    counts cuts, shot_group_id re-identifies recurring shots via ShotMemory. Feed it
    only the frames you keep (blank/frozen frames are not shots) -- Publisher does this,
    and any consumer holding a Gate can drive one directly off (stats, signals):

        tracker = ShotTracker(gate.shot_z, gate.cfg.reid_z)
        stats, signals = gate.frame(frame)
        if not (stats.blank or signals.freeze):
            shot_id, group_id = tracker.update(stats, signals)

    Cut is confirmed one frame late, so at a cut `_last` is the new shot's first frame
    and `_prev` the old shot's last -- the old group is refreshed from `_prev` (drift
    tracking) and the new shot is matched on `_last` (its true first frame)."""

    def __init__(
        self,
        shot_z: Callable[..., float],
        reid_z: float,
        index: Optional[LinearIndex] = None,
    ) -> None:
        self.shot_id = 0
        self.shot_group_id = -1  # -1 until the first shot opens as group 0
        self._mem = ShotMemory(reid_z, index)
        self._score = lambda q, r: shot_z(q.luma, q.color, r.luma, r.color)
        self._last: Optional[ShotRef] = None  # previous kept frame (t-1)
        self._prev: Optional[ShotRef] = None  # kept frame before that (t-2)

    def update(self, stats, signals) -> Tuple[int, int]:
        desc = ShotRef(stats.v_cell_mean, stats.color_mean)
        if self.shot_group_id < 0:  # first shot opens as group 0
            self.shot_group_id, _ = self._mem.match(desc, self._score)
        elif signals.cut:  # cut closes this shot and opens the next
            self.shot_id += 1
            if self._prev is not None:
                self._mem.refresh(self.shot_group_id, self._prev)
            first = self._last if self._last is not None else desc
            self.shot_group_id, _ = self._mem.match(first, self._score)
        self._prev, self._last = self._last, desc
        return self.shot_id, self.shot_group_id

    @property
    def n_groups(self) -> int:
        return len(self._mem)
