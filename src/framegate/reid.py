"""Per-bit Bernoulli shot model for loose re-identification.

A shot is a distribution over hash bits, not a single hash. As its frames stream in
we keep, per bit position, the count of frames in which that bit was set. The bits
that stay constant within a shot (framing, background) become confident; the bits
that flicker (a moving mouth, a talking head) sit near 0.5 and stop discriminating.
Scoring a candidate frame against a shot is then the log-likelihood that its hash was
drawn from that shot's per-bit Bernoullis -- so a re-ID survives exactly the changes
the shot itself already varies over, which is the looseness we want.

The Bernoullis are passed through a binary symmetric channel of flip rate `eps` before
scoring: p -> (1 - 2*eps)*p + eps. Without it a bit seen stable n times drives
log P(flip) to -log n, so the model asserts a bit it has watched for 1000 frames simply
cannot flip and the accept radius shrinks without bound as a shot lengthens -- no value
of the threshold fixes that, because the acceptance region itself is length-dependent.
`eps` is the physical floor: the rate at which codec, resize and sensor noise flip a
bit between two frames of the same setup.

Scoring is a **likelihood ratio**, not a likelihood. An absolute log-likelihood is not
comparable across prototypes of different entropy -- a diffuse one ceilings near log 1/2
for every query, a sharp one scores near 0 -- so a fixed threshold on it silently means
"only near-deterministic prototypes may match". `Background` supplies the null: the bit
distribution of the *population of setups*, one vote per prototype, against which a
candidate is scored. Bits that every setup in the footage shares (a dark surround, one
bright window -- everything shot in the same room) are explained equally well by the null
and contribute ~0; only bits that are stable within a shot AND unusual across the corpus
carry evidence. That is IDF, arrived at rather than bolted on, and it is the standard
GMM-UBM verification score.

Because the score is a *sum* over bits, the null term collapses to one scalar per
(query, group), so `ShotScorer` returns the plain summed log-likelihood and the caller
subtracts `Background.loglik`. Units are total nats, not a per-bit mean: nats compose
with prior odds, and they scale correctly when the code widens beyond one hash.

Everything on the per-frame hot path is integer: a frame is a list append, and the
counts are integer sums. The only floats are the per-bit log-probabilities, and those
are computed once per shot at `finalize()` -- amortised over the whole shot, so still
sub-nanosecond per frame -- not per query. With a Jeffreys prior (a = b = 1/2) the
per-bit probability is (c_i + 1/2) / (n + 1); scoring a query is then a masked add of
the matching per-bit log term over all candidate shots at once.

`ShotProfile` accumulates counts and caches its log terms. `ShotScorer` scores a query
against many profiles in one vectorised pass. `Background` holds the null. None of them
knows about framestore or the pipeline, so all are trivially unit-testable (see
tests/test_reid.py).
"""

from typing import Optional, cast

import numpy as np

BITS = 64  # one uint64 pHash


def _unpack(hashes: np.ndarray) -> np.ndarray:
    """(m,) uint64 -> (m, BITS) uint8, MSB first (imfeat hash bit order)."""
    b = hashes.astype(">u8").view(np.uint8).reshape(-1, 8)
    return np.unpackbits(b, axis=1)


class ShotProfile:
    """Per-bit set-counts for one shot. Frames are buffered and folded into counts in
    one vectorised pass at `finalize()` -- the counts are only read when the shot ends
    (to score the next shot), which is the same cut, so nothing needs them sooner. The
    per-frame cost is therefore a single list append.

    `finalize()` also precomputes the two per-bit log terms the scorer needs, so a
    later score is a masked add with no table lookup -- the scoring hot path never
    touches the counts again."""

    __slots__ = ("key_hash", "n", "counts", "log1", "log0", "p", "eps", "_buf")

    p: Optional[np.ndarray]
    log1: Optional[np.ndarray]
    log0: Optional[np.ndarray]

    def __init__(self, first_hash: int, eps: float = 0.0) -> None:
        self.key_hash = int(first_hash)  # first-frame hash; the framestore key
        self.eps = float(eps)  # BSC flip rate; floors the per-bit log terms
        self.n = 0
        self.counts = np.zeros(BITS, np.int32)
        self.p = None  # shrunk per-bit P(1); the profile's vote in the Background
        self.log1 = self.log0 = None  # log P(bit=1), log P(bit=0), set at finalize
        self._buf: list = []

    def add(self, h: int) -> None:
        self._buf.append(int(h))

    @property
    def pending(self) -> int:
        """Frames buffered but not yet folded into counts."""
        return len(self._buf)

    def fold(self) -> None:
        """Fold buffered frames into the integer counts. Cheap and incremental; called
        both at the closing cut and periodically on a long take to bound the buffer.
        Clears the buffer in place so any bound reference to it (the hot-path cache in
        ShotMemory) stays valid, and drops the cached log terms, which the moved counts
        have just invalidated."""
        if self._buf:
            self.counts += (
                _unpack(np.array(self._buf, np.uint64)).sum(0).astype(np.int32)
            )
            self.n += len(self._buf)
            self._buf.clear()
            self.log1 = self.log0 = self.p = None

    def finalize(self) -> "ShotProfile":
        """Fold any buffered frames, then cache the per-bit log terms the scorer reads.
        Idempotent and cheap on a no-op, so scoring can call it freely: recomputes the
        log terms only when a frame was actually folded in."""
        if not self._buf and self.log1 is not None:
            return self
        self.fold()
        p = (self.counts + 0.5) / (self.n + 1.0)  # Jeffreys-smoothed bit probabilities
        p = (1.0 - 2.0 * self.eps) * p + self.eps  # BSC: keeps log p >= log eps
        self.p = p
        self.log1 = np.log(p)
        self.log0 = np.log(1.0 - p)
        return self

    @property
    def dist(self) -> np.ndarray:
        """Finalized per-bit P(1) -- this prototype's single vote in the Background."""
        return cast(np.ndarray, self.finalize().p)


class ShotScorer:
    """Scores a query hash against a batch of finalized profiles. The per-bit log terms
    live on the profiles (computed at finalize), so this is a single masked add over
    the (K, BITS) stack -- no table, no per-shot float work here."""

    def score(self, q_hash: int, profiles: list) -> np.ndarray:
        """Total log-likelihood (nats) of `q_hash` under each profile, as a (K,) array.
        Subtract `Background.loglik` for the group to turn it into the likelihood ratio
        that is actually compared against a threshold."""
        if not profiles:
            return np.empty(0, np.float64)
        q = _unpack(np.array([q_hash], np.uint64))[0].astype(bool)  # (BITS,)
        log1 = np.stack([p.log1 for p in profiles])  # (K, BITS) log P(bit=1)
        log0 = np.stack([p.log0 for p in profiles])
        return np.where(q, log1, log0).sum(1)


class Background:
    """The null model: "this frame is some setup, just not that one".

    One vote per prototype, never per frame. Frame-weighting would let a single long
    take become the population -- and the setup it destroys is its own, since its
    evidence against a null that is a copy of itself falls to ~0.

    A group is also left out of its own null (cohort normalisation / T-norm). Early in
    a video a group is a large fraction of the population it is scored against, which
    halves its own evidence; with one group seen the null would be an exact copy of the
    only candidate and nothing could ever match. Leaving it out is a subtraction off the
    running sum, so it costs nothing, and the empty-population limit falls out as the
    uninformative null p = 1/2."""

    __slots__ = ("eps", "_sum", "_k", "_gsum", "_gk", "_last", "_cache")

    def __init__(self, eps: float = 0.0) -> None:
        self.eps = float(eps)
        self._sum = np.zeros(BITS)  # sum of prototype bit distributions
        self._k = 0  # prototypes voting
        self._gsum: dict = {}  # gid -> that group's partial sum
        self._gk: dict = {}
        self._last: dict = {}  # prototype key -> its last counted vote
        self._cache: dict = {}

    def update(self, key: int, gid: int, p: np.ndarray) -> None:
        """Register a prototype's current bit distribution. Idempotent per key: a
        prototype that has folded more frames replaces its own earlier vote."""
        last = self._last.get(key)
        if last is None:
            self._k += 1
            self._gk[gid] = self._gk.get(gid, 0) + 1
            self._gsum.setdefault(gid, np.zeros(BITS))
        else:
            self._sum -= last
            self._gsum[gid] -= last
        self._sum += p
        self._gsum[gid] += p
        self._last[key] = p
        self._cache.clear()

    def loglik(self, q_hash: int, gid: Optional[int] = None) -> float:
        """Total log-likelihood (nats) of `q_hash` under the null, with `gid` left out."""
        lo = self._cache.get(gid)
        if lo is None:
            s, k = self._sum, self._k
            if gid in self._gk:
                s, k = s - self._gsum[gid], k - self._gk[gid]
            g = (s + 0.5) / (k + 1.0)  # k = 0 gives the uniform null, p = 1/2
            g = np.clip(g, self.eps, 1.0 - self.eps)  # cap how much one rare bit is worth
            lo = self._cache[gid] = (np.log(g), np.log(1.0 - g))
        q = _unpack(np.array([q_hash], np.uint64))[0].astype(bool)
        return float(np.where(q, lo[0], lo[1]).sum())

    def __len__(self) -> int:
        return self._k
