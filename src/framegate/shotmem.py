"""Shot re-identification (L2 memory): recognise when a shot re-appears -- the
wide / A / B setups a dialogue or cross-cut keeps returning to -- so each shot gets a
unique `shot_id` and a `shot_group_id` shared by every recurrence of the same setup.

Two stages, cheap-then-precise. A framestore index over luma pHashes gives the recall
net: a new shot's first-frame hash retrieves candidates within a generous Hamming
radius in microseconds. Each candidate is then scored by the per-bit Bernoulli model
(reid.ShotScorer) against the null of reid.Background -- the log-likelihood *ratio*
that the frame was drawn from that candidate rather than from the population of setups
seen so far. Those ratios are then turned into a posterior over "which group, or a new
one" under a Chinese-restaurant prior, and the winner is a re-ID iff its posterior
log-odds clear `reid_llr` nats. Retrieval is loose
on purpose; precision comes from scoring, which is what lets the same setup re-ID across
the pose and speech changes a talking head goes through (see reid.py). framestore stays
a pure Hamming index; all the probability lives here.

A group is held as one or more *prototypes*, each a (framestore key, local Bernoulli
profile). A near-static shot is a single prototype; a shot that drifts (a pan, a zoom)
spawns more, so both recall and scoring stay tight to the part of the shot they cover
instead of blurring into one shot-wide average. A group's score is the *mixture* over
its prototypes, not the best of them: taking the max gives a group one independent
attempt at the threshold per prototype, so a group that has spread over a long pan
out-competes a tight one for no reason beyond having more shapes to try.

`ShotMemory` owns the store, the groups and their prototypes, and drives the matching.
`ShotTracker` drives it from the per-frame gate outputs, accumulating the current
shot's frames and closing / opening groups at each cut.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import framestore  # type: ignore[import-untyped]  # framestore needs a py.typed marker

from .reid import Background, ShotProfile, ShotScorer

# Pseudocount on the mixture weights. A prototype spawned by drift has never opened a
# shot, and weight 0 would assert that a recurrence can only open where the group first
# did -- which is exactly what a shot that pans away and comes back does not do. One
# pseudocount keeps every prototype viable while opening statistics accumulate.
OPEN_PRIOR = 1.0


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()  # requires Python 3.10+ (see requires-python)


def _logsumexp(x: np.ndarray) -> float:
    m = float(x.max())
    return m + float(np.log(np.exp(x - m).sum()))


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
    slot: int = -1  # framestore id; also this prototype's key in the Background
    opens: int = 0  # shots that opened on this prototype; its mixture weight


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
    group (posterior log-odds >= reid_llr) or creates a new one. `accumulate(gid, hash)` folds a frame
    into the nearest prototype of its group, spawning a new prototype when the frame has
    drifted past reid_maxd from all existing ones."""

    def __init__(
        self,
        reid_maxd: float,
        reid_llr: float,
        reid_eps: float = 0.0,
        reid_alpha: float = 1.0,
    ) -> None:
        self._maxd = reid_maxd
        self._eps = reid_eps
        self._maxbits = int(reid_maxd * 64)  # recall radius in bits
        # A prototype's profile covers frames within `spawn` bits of its key, so a new
        # prototype spawns at half the recall radius: the clouds overlap and no frame
        # lands in a scoring dead zone between two centres, while recall still reaches
        # anything within the full radius.
        self._spawnbits = max(1, self._maxbits // 2)
        self._llr = reid_llr
        self._store = framestore.Store()
        self._scorer = ShotScorer()
        self._bg = Background(reid_eps)
        self._logalpha = float(np.log(reid_alpha))
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
        gid, pr, _ = m
        pr.opens += 1
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
        pr.slot = int(self._store.insert(np.array([h], np.uint64))[0])
        self._by_slot[pr.slot] = pr
        return pr

    def _match(self, q_hash: int) -> Optional[Tuple[int, "Prototype", float]]:
        """Best (group, prototype, log-likelihood ratio) for `q_hash`, or None if no
        group clears `reid_llr`."""
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
        lls = self._scorer.score(int(q_hash), [pr.profile.finalize() for pr in protos])
        cand: Dict[int, List[Tuple[float, Prototype]]] = {}
        for pr, ll in zip(protos, lls):  # mixture terms: log w_k + log P(q | k)
            cand.setdefault(pr.group_id, []).append((float(ll) + self._logw(pr), pr))
        # Weights normalise over ALL of a group's prototypes, so the ones retrieval did
        # not return are correctly absent rather than silently reweighted, and a group
        # spread over many pays the log K it owes. The null is constant within a group,
        # so it comes off the mixture once.
        gids, best, logits = [], [], []
        for gid, terms in cand.items():
            gids.append(gid)
            best.append(max(terms, key=lambda t: t[0])[1])
            logits.append(
                np.log(self._n_shots(gid)) + _logsumexp(np.array([t[0] for t in terms]))
            )

        # Posterior over "which group, or a new one" under a Chinese-restaurant prior:
        #   P(g|q) ~ n_g P(q|g),   P(new|q) ~ alpha P(q|population).
        # The prior normaliser cancels, and groups retrieval did not return have
        # P(q|.) ~ 0, so the competition is over the candidates. Note the background
        # appears exactly ONCE, in the new-setup branch. Dividing every term by its own
        # leave-one-out null instead would make the terms incommensurable and count a
        # rival twice -- once inside the candidate's null and again as a competitor --
        # which on an exactly ambiguous query inflates the winner rather than blocking
        # it. Competition belongs in the denominator; the null belongs to "new".
        arr = np.array(logits)
        i = int(arr.argmax())
        rest = _logsumexp(
            np.append(
                np.delete(arr, i), self._logalpha + self._bg.loglik(q_hash, gids[i])
            )
        )
        odds = float(arr[i] - rest)
        return (gids[i], best[i], odds) if odds >= self._llr else None

    def _n_shots(self, gid: int) -> int:
        """Shots this group has held. Every shot opening lands on exactly one prototype
        and increments its `opens`, so the group's total is their sum -- no separate
        counter to keep consistent."""
        return max(1, sum(pr.opens for pr in self._groups[gid].prototypes))

    def _logw(self, pr: Prototype) -> float:
        """Log mixture weight of a prototype: the share of its group's shot openings
        that landed on it. The query is a shot-opening frame, so opening counts are the
        right statistic -- frame mass would ask instead which prototype a *random*
        frame of the group resembles, and a shot that opens on one framing and then
        pans for a thousand frames puts almost all its mass somewhere it never opens."""
        g = self._groups[pr.group_id]
        z = sum(q.opens + OPEN_PRIOR for q in g.prototypes)
        return float(np.log((pr.opens + OPEN_PRIOR) / z))

    def _new_group(self, first_hash: int) -> int:
        gid = self._n
        self._n += 1
        h = int(first_hash)
        pr = Prototype(h, gid, ShotProfile(h, self._eps), opens=1)
        self._groups[gid] = Group(gid, [pr])
        self._set_hot(gid, pr)
        pr.slot = int(self._store.insert(np.array([h], np.uint64))[0])
        self._by_slot[pr.slot] = pr
        return gid

    def finalize_group(self, gid: int) -> None:
        """Fold every prototype of a group and refresh its vote in the null -- called
        when a shot closes, so its frames are counted before the next match scores
        against them and the population they widen."""
        for pr in self._groups[gid].prototypes:
            self._bg.update(pr.slot, gid, pr.profile.dist)

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
        self._mem = ShotMemory(
            cfg.reid_maxd, cfg.reid_llr, cfg.reid_eps, cfg.reid_alpha
        )
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
