"""Shot re-identification (L2 memory): recognise when a shot re-appears -- the
wide / A / B setups a dialogue or cross-cut keeps returning to -- so each shot gets a
unique `shot_id` and a `shot_group_id` shared by every recurrence of the same setup.

Two stages, cheap-then-precise. A framestore index over luma pHashes gives the recall
net: a new shot's first-frame hash retrieves candidates within a generous Hamming
radius in microseconds. Each candidate is then scored by the per-bit Bernoulli model
(reid.ShotScorer) -- the mean log-likelihood that the frame was drawn from that
candidate's learned bit distribution -- and the best-scoring group above `reid_ll`
wins. Retrieval is loose on purpose; precision comes from scoring, which is what lets
the same setup re-ID across the pose and speech changes a talking head goes through
(see reid.py). framestore stays a pure Hamming index; all the probability lives here.

A group is held as one or more *prototypes*, each a (framestore key, local Bernoulli
profile). A near-static shot is a single prototype; a shot that drifts (a pan, a zoom)
spawns more, so both recall and scoring stay tight to the part of the shot they cover
instead of blurring into one shot-wide average.

`ShotMemory` owns the store, the groups and their prototypes, and drives the matching.
`ShotTracker` drives it from the per-frame gate outputs, accumulating the current
shot's frames and closing / opening groups at each cut.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import framestore  # type: ignore[import-untyped]  # framestore needs a py.typed marker

from .reid import ShotProfile, ShotScorer


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()  # requires Python 3.10+ (see requires-python)


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
class Prototype:
    """One local appearance within a group: a framestore key and the Bernoulli profile
    of the frames that landed nearest it. A shot that drifts (a pan, a zoom) spawns
    several, so both recall (the key) and scoring (the local profile) stay tight to the
    part of the shot they cover, instead of blurring into one shot-wide average."""

    key: int
    group_id: int
    profile: ShotProfile


@dataclass
class Group:
    """A re-identifiable setup, held as one or more local prototypes. It starts with a
    single prototype (the first frame); a frame that drifts past `reid_maxd` from every
    existing prototype's key spawns a new one. Recall retrieves prototypes by key; a
    group's match score is the best over its prototypes, so the drifted end of a shot
    matches its own local prototype rather than the shot-wide mean."""

    group_id: int
    prototypes: List[Prototype]
    member_shot_ids: List[int] = field(default_factory=list)


class ShotMemory:
    """Assigns shot group ids by loose pHash re-identification. `open_shot(hash)` at a
    cut retrieves candidate prototypes from the store, scores the query against each
    one's local Bernoulli profile, takes the best per group, and either re-IDs the best
    group (score >= reid_ll) or creates a new one. `accumulate(gid, hash)` folds a frame
    into the nearest prototype of its group, spawning a new prototype when the frame has
    drifted past reid_maxd from all existing ones."""

    def __init__(self, reid_maxd: float, reid_ll: float, reid_eps: float = 0.0) -> None:
        self._maxd = reid_maxd
        self._eps = reid_eps
        self._maxbits = int(reid_maxd * 64)  # recall radius in bits
        # A prototype's profile covers frames within `spawn` bits of its key, so a new
        # prototype spawns at half the recall radius: the clouds overlap and no frame
        # lands in a scoring dead zone between two centres, while recall still reaches
        # anything within the full radius.
        self._spawnbits = max(1, self._maxbits // 2)
        self._ll = reid_ll
        self._store = framestore.Store()
        self._scorer = ShotScorer()
        self._groups: Dict[int, Group] = {}
        self._by_slot: Dict[int, Prototype] = {}  # framestore id -> prototype
        # Per-group hot cache: the active prototype pre-unpacked into the four things
        # accumulate touches, so the frame path walks no attributes and calls no
        # property. Rebuilt only when the active prototype changes (on drift).
        self._hot: Dict[int, Tuple[int, Any, list, ShotProfile]] = {}
        self._n = 0

    def open_shot(self, first_hash: int) -> Tuple[int, bool]:
        """Return (group_id, is_new) for a shot opening on `first_hash`."""
        m = self._match(first_hash)
        if m is None:
            return self._new_group(first_hash), True
        gid, pr = m
        self._set_hot(gid, pr)  # the matched prototype is the one this shot resumes on
        return gid, False

    def _set_hot(self, gid: int, pr: Prototype) -> None:
        buf = pr.profile._buf
        self._hot[gid] = (pr.key, buf.append, buf, pr.profile)

    def accumulate(self, gid: int, frame_hash: int) -> None:
        """Fold a frame into the active prototype of its group. The hot path is a single
        popcount against the prototype's key and one list append. Only a frame that
        drifts past the spawn radius (rare) leaves it, to search for or spawn a nearer
        prototype."""
        key, append, buf, profile = self._hot[gid]
        if (frame_hash ^ key).bit_count() <= self._spawnbits:
            append(frame_hash)
            if len(buf) >= 4096:  # bound an open prototype's buffer on very long takes
                profile.fold()
            return
        self._drift(gid, int(frame_hash))

    def _drift(self, gid: int, h: int) -> None:
        pr = self._nearest_or_new(self._groups[gid], h)
        pr.profile.add(h)
        self._set_hot(gid, pr)

    def _nearest_or_new(self, g: Group, h: int) -> Prototype:
        best, best_d = g.prototypes[0], _hamming(h, g.prototypes[0].key)
        for pr in g.prototypes[1:]:
            d = _hamming(h, pr.key)
            if d < best_d:
                best, best_d = pr, d
        if best_d <= self._spawnbits:
            return best
        return self._add_prototype(g, h)  # drifted past every prototype -> new one

    def _add_prototype(self, g: Group, h: int) -> Prototype:
        pr = Prototype(h, g.group_id, ShotProfile(h, self._eps))
        g.prototypes.append(pr)
        slot = int(self._store.insert(np.array([h], np.uint64))[0])
        self._by_slot[slot] = pr
        return pr

    def _match(self, q_hash: int) -> Optional[Tuple[int, "Prototype"]]:
        if self._n == 0:
            return None
        _, ids = self._store.query(int(q_hash), k=None, max_dist=self._maxd)
        seen: Dict[int, Prototype] = {}
        for (
            s
        ) in ids:  # several keys of one prototype can be returned; dedup by identity
            pr = self._by_slot[int(s)]
            seen.setdefault(id(pr), pr)
        if not seen:
            return None
        protos = list(seen.values())
        scores = self._scorer.score(
            int(q_hash), [pr.profile.finalize() for pr in protos]
        )
        best: Dict[int, Tuple[float, Prototype]] = {}
        for pr, sc in zip(protos, scores):  # a group's score is its best prototype
            if sc > best.get(pr.group_id, (-np.inf, pr))[0]:
                best[pr.group_id] = (float(sc), pr)
        gid, (sc, pr) = max(best.items(), key=lambda kv: kv[1][0])
        return (gid, pr) if sc >= self._ll else None

    def _new_group(self, first_hash: int) -> int:
        gid = self._n
        self._n += 1
        h = int(first_hash)
        pr = Prototype(h, gid, ShotProfile(h, self._eps))
        self._groups[gid] = Group(gid, [pr])
        self._set_hot(gid, pr)
        slot = int(self._store.insert(np.array([h], np.uint64))[0])
        self._by_slot[slot] = pr
        return gid

    def finalize_group(self, gid: int) -> None:
        """Fold every prototype of a group -- called when a shot closes so its frames
        are counted before the next match scores against them."""
        for pr in self._groups[gid].prototypes:
            pr.profile.finalize()

    def prototypes(self, gid: int) -> List[Prototype]:
        return self._groups[gid].prototypes

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

    Cut is confirmed one frame late, so at a cut the *pending* frame (t-1) is the new
    shot's first. Accumulation therefore lags one frame behind the stream: a frame is
    folded into a group's profile only once the next frame has said which shot it
    belongs to. Otherwise every cut folds the new shot's first frame -- the one frame
    that is guaranteed to belong to a different setup -- into the outgoing group.
    Every kept frame's hash is folded into its group's profile, so a group's Bernoulli
    distribution reflects all recurrences of the setup, not just its first."""

    def __init__(self, cfg) -> None:
        self.shot_id = 0
        self.shot_group_id = -1  # -1 until the first shot opens as group 0
        self._mem = ShotMemory(cfg.reid_maxd, cfg.reid_ll, cfg.reid_eps)
        self._pending: Optional[Tuple[int, int]] = None  # (hash, frame_id) at t-1
        self._shots: List[Shot] = []
        self._start_frame = 0
        self._end_frame = 0
        self._n_frames = 0  # frames in the CURRENT shot (group profile.n is cumulative)

    def update(self, stats, signals, frame_id: int = -1) -> Tuple[int, int]:
        pend = self._pending
        if pend is None:  # first kept frame ever opens shot 0
            self.shot_group_id, _ = self._mem.open_shot(int(stats.phash))
            self._start_frame = frame_id
        elif signals.cut:  # the pending frame is the new shot's first, not the old's
            self._close_shot()
            self.shot_id += 1
            self.shot_group_id, _ = self._mem.open_shot(pend[0])
            self._start_frame = pend[1]
            self._absorb(pend)
        else:
            self._absorb(pend)
        self._pending = (int(stats.phash), frame_id)
        return self.shot_id, self.shot_group_id

    def _absorb(self, pend: Tuple[int, int]) -> None:
        self._mem.accumulate(self.shot_group_id, pend[0])
        self._n_frames += 1
        self._end_frame = pend[1]

    def _close_shot(self) -> None:
        gid = self.shot_group_id
        self._mem.finalize_group(gid)
        self._mem.group(gid).member_shot_ids.append(self.shot_id)
        self._shots.append(
            Shot(
                shot_id=self.shot_id,
                group_id=gid,
                start_frame=self._start_frame,
                end_frame=self._end_frame,
                n_frames=self._n_frames,
            )
        )
        self._n_frames = 0

    @property
    def shots(self) -> List[Shot]:
        return list(self._shots)

    @property
    def n_groups(self) -> int:
        return len(self._mem)
