"""Shot re-identification (L2 memory): recognise when a shot re-appears -- the
wide / A / B setups a dialogue or cross-cut keeps returning to -- so each shot gets a
unique `shot_id` and a `shot_group_id` shared by every recurrence of the same setup.

Two stages, cheap-then-precise. A framestore index over one uint64 pHash per stored
shot gives the recall net: a new shot's first-frame hash retrieves candidate groups
within a generous Hamming radius in microseconds. Each candidate is then scored by the
per-bit Bernoulli model (reid.ShotScorer) -- the mean log-likelihood that the frame was
drawn from that group's learned bit distribution -- and the best above `reid_ll` wins.
The retrieval is loose on purpose and the precision comes from scoring, which is what
lets the same setup re-ID across the pose and speech changes a talking head goes
through (see reid.py). framestore stays a pure Hamming index; all the probability lives
here.

`ShotMemory` owns the store, the per-group `ShotProfile`s, and the per-shot metadata.
`ShotTracker` drives it from the per-frame gate outputs, accumulating the current
shot's frames and closing / opening groups at each cut.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import framestore  # type: ignore[import-untyped]  # framestore needs a py.typed marker

from .reid import ShotProfile, ShotScorer


@dataclass
class Shot:
    """One continuous run between cuts. `scene_graph` is a reserved slot for a later
    per-shot annotation (OCR, detections, ...); unused for now."""

    shot_id: int
    group_id: int
    start_frame: int
    end_frame: int
    n_frames: int
    scene_graph: object = None


@dataclass
class Group:
    """A re-identifiable setup: its framestore key (first-frame hash), the Bernoulli
    profile accumulated over every frame ever assigned to it, and the shots that are
    recurrences of it."""

    group_id: int
    profile: ShotProfile
    member_shot_ids: List[int] = field(default_factory=list)


class ShotMemory:
    """Assigns shot group ids by loose pHash re-identification. `open_shot(hash)` at a
    cut retrieves candidate groups from the store, scores the first frame against each,
    and either re-IDs the best (score >= reid_ll) or creates a new group keyed on that
    hash. `accumulate(gid, hash)` folds a frame into a group's profile."""

    def __init__(self, reid_maxd: float, reid_ll: float) -> None:
        self._maxd = reid_maxd
        self._ll = reid_ll
        self._store = framestore.Store()
        self._scorer = ShotScorer()
        self._groups: Dict[int, Group] = {}
        self._by_slot: Dict[int, int] = {}  # framestore id -> group_id
        self._n = 0

    def open_shot(self, first_hash: int) -> Tuple[int, bool]:
        """Return (group_id, is_new) for a shot opening on `first_hash`."""
        gid = self._match(first_hash)
        if gid is None:
            gid = self._new_group(first_hash)
            return gid, True
        return gid, False

    def accumulate(self, gid: int, frame_hash: int) -> None:
        p = self._groups[gid].profile
        p.add(frame_hash)
        if p.pending >= 4096:  # bound the open-shot buffer on very long takes
            p.fold()

    def _match(self, q_hash: int) -> Optional[int]:
        if self._n == 0:
            return None
        _, ids = self._store.query(int(q_hash), k=None, max_dist=self._maxd)
        cand = [self._by_slot[int(s)] for s in ids]
        if not cand:
            return None
        profs = [self._groups[g].profile.finalize() for g in cand]
        scores = self._scorer.score(int(q_hash), profs)
        best = int(np.argmax(scores))
        return cand[best] if scores[best] >= self._ll else None

    def _new_group(self, first_hash: int) -> int:
        gid = self._n
        self._n += 1
        self._groups[gid] = Group(gid, ShotProfile(first_hash))
        slot = int(self._store.insert(np.array([first_hash], np.uint64))[0])
        self._by_slot[slot] = gid
        return gid

    def profile(self, gid: int) -> ShotProfile:
        return self._groups[gid].profile

    def group(self, gid: int) -> Group:
        return self._groups[gid]

    def __len__(self) -> int:
        return self._n


class ShotTracker:
    """Turns a stream of per-frame gate outputs into (shot_id, shot_group_id): shot_id
    counts cuts, shot_group_id re-identifies recurring setups. Feed it only the frames
    you keep (blank/frozen frames are not shots) -- Publisher does this, and any
    consumer holding a Gate can drive one off (stats, signals):

        tracker = ShotTracker(gate.cfg)
        stats, signals = gate.frame(frame)
        if not (stats.blank or signals.freeze):
            shot_id, group_id = tracker.update(stats, signals, frame_id)

    Cut is confirmed one frame late, so at a cut `_last_hash` is the new shot's first
    frame. Every kept frame's hash is folded into its group's profile, so a group's
    Bernoulli distribution reflects all recurrences of the setup, not just its first."""

    def __init__(self, cfg) -> None:
        self.shot_id = 0
        self.shot_group_id = -1  # -1 until the first shot opens as group 0
        self._mem = ShotMemory(cfg.reid_maxd, cfg.reid_ll)
        self._last_hash: Optional[int] = None  # previous kept frame's hash (t-1)
        self._shots: List[Shot] = []
        self._start_frame = 0
        self._n_frames = 0  # frames in the CURRENT shot (group profile.n is cumulative)

    def update(self, stats, signals, frame_id: int = -1) -> Tuple[int, int]:
        h = int(stats.phash)
        if self.shot_group_id < 0:  # first shot opens as group 0
            self.shot_group_id, _ = self._mem.open_shot(h)
            self._start_frame = frame_id
            self._n_frames = 0
        elif signals.cut:  # cut closes this shot and opens the next
            self._close_shot(frame_id)
            self.shot_id += 1
            first = self._last_hash if self._last_hash is not None else h
            self.shot_group_id, _ = self._mem.open_shot(first)
            self._start_frame = frame_id
            self._n_frames = 0
        self._mem.accumulate(self.shot_group_id, h)
        self._n_frames += 1
        self._last_hash = h
        return self.shot_id, self.shot_group_id

    def _close_shot(self, end_frame: int) -> None:
        gid = self.shot_group_id
        self._mem.profile(gid).finalize()
        self._mem.group(gid).member_shot_ids.append(self.shot_id)
        self._shots.append(
            Shot(
                shot_id=self.shot_id,
                group_id=gid,
                start_frame=self._start_frame,
                end_frame=end_frame,
                n_frames=self._n_frames,
            )
        )

    @property
    def shots(self) -> List[Shot]:
        return list(self._shots)

    @property
    def n_groups(self) -> int:
        return len(self._mem)
