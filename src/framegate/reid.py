"""Per-bit Bernoulli shot model for loose re-identification.

A shot is a distribution over hash bits, not a single hash. As its frames stream in
we keep, per bit position, the count of frames in which that bit was set. The bits
that stay constant within a shot (framing, background) become confident; the bits
that flicker (a moving mouth, a talking head) sit near 0.5 and stop discriminating.
Scoring a candidate frame against a shot is then the log-likelihood that its hash was
drawn from that shot's per-bit Bernoullis -- so a re-ID survives exactly the changes
the shot itself already varies over, which is the looseness we want.

Everything on the per-frame hot path is integer: a frame is a list append, and the
counts are integer sums. The only floats are the per-bit log-probabilities, and those
are computed once per shot at `finalize()` -- amortised over the whole shot, so still
sub-nanosecond per frame -- not per query. With a Jeffreys prior (a = b = 1/2) the
per-bit probability is (c_i + 1/2) / (n + 1); scoring a query is then a masked add of
the matching per-bit log term over all candidate shots at once. The score is the mean
per-bit log-likelihood, so the threshold is independent of the hash width.

`ShotProfile` accumulates counts and caches its log terms. `ShotScorer` scores a query
against many profiles in one vectorised pass. Neither knows about framestore or the
pipeline, so both are trivially unit-testable (see tests/test_reid.py).
"""

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

    __slots__ = ("key_hash", "n", "counts", "log1", "log0", "denom", "_buf")

    def __init__(self, first_hash: int) -> None:
        self.key_hash = int(first_hash)  # first-frame hash; the framestore key
        self.n = 0
        self.counts = np.zeros(BITS, np.int32)
        self.log1 = self.log0 = None  # log P(bit=1), log P(bit=0), set at finalize
        self.denom = 0.0
        self._buf: list = []

    def add(self, h: int) -> None:
        self._buf.append(int(h))

    def finalize(self) -> "ShotProfile":
        """Fold buffered frames into counts and cache the per-bit log terms. Idempotent
        across repeated calls (re-folds only what is newly buffered)."""
        if self._buf:
            self.counts += (
                _unpack(np.array(self._buf, np.uint64)).sum(0).astype(np.int32)
            )
            self.n += len(self._buf)
            self._buf = []
        p = (self.counts + 0.5) / (self.n + 1.0)  # Jeffreys-smoothed bit probabilities
        self.log1 = np.log(p)
        self.log0 = np.log(1.0 - p)
        return self


class ShotScorer:
    """Scores a query hash against a batch of finalized profiles. The per-bit log terms
    live on the profiles (computed at finalize), so this is a single masked add over
    the (K, BITS) stack -- no table, no per-shot float work here."""

    def score(self, q_hash: int, profiles: list) -> np.ndarray:
        """Mean per-bit log-likelihood of `q_hash` under each profile, as a (K,) array.
        Higher is a better match; range is roughly [-log 2, 0]."""
        k = len(profiles)
        if k == 0:
            return np.empty(0, np.float64)
        q = _unpack(np.array([q_hash], np.uint64))[0].astype(bool)  # (BITS,)
        log1 = np.stack([p.log1 for p in profiles])  # (K, BITS) log P(bit=1)
        log0 = np.stack([p.log0 for p in profiles])
        return np.where(q, log1, log0).mean(1)
