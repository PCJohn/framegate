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
