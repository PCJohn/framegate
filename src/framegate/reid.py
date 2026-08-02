"""Per-bit Bernoulli shot model and the population null it is scored against.

A shot is a distribution over hash bits, not a single hash. As its frames stream in we
keep, per bit position, the count of frames in which that bit was set. Bits that stay
constant within a shot (framing, background) become confident; bits that flicker (a
moving mouth) sit near 0.5 and stop discriminating. A re-ID therefore survives exactly
the changes the shot itself already varies over, which is the looseness we want.

Three things shape the score, each fixing a specific failure:

**Bounded emission.** The Bernoullis pass through a binary symmetric channel of flip
rate `eps`: p -> (1 - 2*eps)*p + eps. Without it a bit seen stable n times drives
log P(flip) to -log n, so the model asserts a bit it has watched for 1000 frames cannot
flip, and the accept radius shrinks without bound as a shot lengthens. No threshold
value fixes that, because the acceptance region itself is length-dependent. `eps` is not
a noise parameter to fit -- it is a contamination floor asserting that no bit is ever
certain, the role variance flooring plays in GMM-UBM.

**A likelihood ratio, not a likelihood.** An absolute log-likelihood is not comparable
across profiles of different entropy: a diffuse one ceilings near log 1/2 for every
query, a sharp one scores near 0, so a fixed threshold on it silently means "only
near-deterministic profiles may match". `Background` supplies the null -- the bit
distribution of the *population of setups*. Bits every setup in the footage shares (a
dark surround, one bright window: everything shot in one room) are explained equally
well by the null and stop carrying evidence; what survives is stable within the shot AND
unusual across the corpus. That is IDF, arrived at rather than bolted on, and it is the
standard GMM-UBM verification score.

**One vote per prototype, and leave-one-out.** Frame-weighting the null would let a
single long take become the population, and the setup it destroys first is its own,
since its evidence against a copy of itself falls to ~0. Leaving a group out of the null
matters early, when a group is a large fraction of the population it is measured
against. See `Background`.

Everything scores as a **linear function of the query bits**. Writing q for the 0/1 bit
vector,

    log P(q | p) = sum_i log(1 - p_i)  +  sum_i q_i * log(p_i / (1 - p_i))
                 = const + logit . q

so a profile reduces to a scalar and a 64-vector of log-odds weights, and scoring a
candidate is one dot product. Subtracting a null is then just subtracting its
`const` and `logit`: the whole re-ID decision is a linear classifier over hash bits whose
weights are learned per shot. Both are computed once per shot at `finalize()`, never per
query.

The per-frame hot path stays integer: a frame is a list append, and the counts are
integer sums. Floats appear only when a shot closes.

Nothing here knows about framestore or the pipeline, so it is all trivially
unit-testable (see tests/test_reid.py). `shotmem.py` composes them into the decision.
"""

import numpy as np

BITS = 64  # one uint64 pHash


def _unpack(hashes: np.ndarray) -> np.ndarray:
    """(m,) uint64 -> (m, BITS) uint8, MSB first (imfeat hash bit order)."""
    b = hashes.astype(">u8").view(np.uint8).reshape(-1, 8)
    return np.unpackbits(b, axis=1)


def bits(h: int) -> np.ndarray:
    """One hash -> the (BITS,) float 0/1 vector every score dots against. Unpacked once
    per query and passed around; it is the only per-query allocation on the cut path."""
    return np.unpackbits(np.frombuffer(h.to_bytes(8, "big"), np.uint8)).astype(
        np.float64
    )


def _linear(p: np.ndarray) -> tuple[float, np.ndarray]:
    """Bit probabilities -> (const, logit), so that log P(q|p) == const + logit @ q."""
    return float(np.log1p(-p).sum()), np.log(p) - np.log1p(-p)


class ShotProfile:
    """Per-bit set-counts for one prototype. Frames are buffered and folded into the
    counts in one vectorised pass at `finalize()` -- nothing reads them until the shot
    ends, which is exactly when the next match needs them -- so the per-frame cost is a
    single list append.

    `finalize()` also caches the (const, logit) pair the scorer reads, so scoring never
    touches the counts. It is idempotent and cheap on a no-op, so `score()` calls it
    itself and callers never have to remember to."""

    __slots__ = (
        "_buf",
        "_stale",
        "const",
        "counts",
        "eps",
        "key_hash",
        "logit",
        "n",
        "p",
    )

    def __init__(self, first_hash: int, eps: float = 0.0) -> None:
        self.key_hash = int(first_hash)  # first-frame hash; the framestore key
        self.eps = float(eps)  # BSC flip rate; floors the per-bit log terms
        self.n = 0
        self.counts = np.zeros(BITS, np.int32)
        self.p = np.full(BITS, 0.5)  # uniform until the first fold
        self.const, self.logit = _linear(self.p)
        self._stale = True
        self._buf: list[int] = []

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
        ShotMemory) stays valid, and drops the cached derivation, which the moved counts
        have just invalidated."""
        if self._buf:
            self.counts += (
                _unpack(np.array(self._buf, np.uint64)).sum(0).astype(np.int32)
            )
            self.n += len(self._buf)
            self._buf.clear()
            self._stale = True

    def finalize(self) -> "ShotProfile":
        """Fold any buffered frames, then cache what the scorer reads. Idempotent and
        cheap on a no-op, so scoring can call it freely."""
        if not (self._stale or self._buf):
            return self
        self.fold()
        p = (self.counts + 0.5) / (self.n + 1.0)  # Jeffreys-smoothed bit probabilities
        self.p = (1.0 - 2.0 * self.eps) * p + self.eps  # BSC: keeps log p >= log eps
        self.const, self.logit = _linear(self.p)
        self._stale = False
        return self

    @property
    def dist(self) -> np.ndarray:
        """Finalized per-bit P(1) -- this prototype's single vote in the Background."""
        return self.finalize().p


def score(q_bits: np.ndarray, profiles: list["ShotProfile"]) -> list[float]:
    """Total log-likelihood (nats) of the query under each profile.

    Deliberately a Python loop rather than a stacked matrix-vector product: candidate
    lists are a handful long, and stacking costs more than it saves below ~16 profiles
    (measured 1.0 us vs 4.7 us at K=1, crossing over near K=16). Profiles are finalized
    here, so callers never have to remember to."""
    return [p.finalize().const + float(p.logit @ q_bits) for p in profiles]


class Background:
    """The null: "this frame is some setup, just not that one".

    One vote per prototype, never per frame -- see the module docstring for why frame
    weighting is self-defeating. Held as a running sum of the prototypes' bit
    distributions, with a per-group partial sum so a group can be left out of its own
    null (cohort normalisation / T-norm). That matters early: with few groups known a
    candidate is a large fraction of the population it is scored against, and with one
    group the null would be a copy of the only candidate. Leaving it out is a
    subtraction off the running sum, so it is free, and the empty-population limit falls
    out as the uninformative null p = 1/2.

    The null enters the decision exactly ONCE, as the "or a new setup" branch of the
    posterior -- not as a per-candidate divisor. See `ShotMemory._match`."""

    __slots__ = ("_cache", "_gk", "_gsum", "_k", "_last", "_sum", "eps")

    def __init__(self, eps: float = 0.0) -> None:
        self.eps = float(eps)
        self._sum = np.zeros(BITS)  # sum of prototype bit distributions
        self._k = 0  # prototypes voting
        self._gsum: dict[int, np.ndarray] = {}  # gid -> that group's partial sum
        self._gk: dict[int, int] = {}
        self._last: dict[int, np.ndarray] = {}  # prototype key -> its last counted vote
        self._cache: dict[int | None, tuple[float, np.ndarray]] = {}

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

    def loglik(self, q_bits: np.ndarray, gid: int | None = None) -> float:
        """Total log-likelihood (nats) of the query under the null, `gid` left out."""
        lin = self._cache.get(gid)
        if lin is None:
            s, k = self._sum, self._k
            if gid in self._gk:
                s, k = s - self._gsum[gid], k - self._gk[gid]
            g = (s + 0.5) / (k + 1.0)  # k = 0 gives the uniform null, p = 1/2
            g = np.clip(g, self.eps, 1.0 - self.eps)  # cap what one rare bit is worth
            lin = self._cache[gid] = _linear(g)
        return lin[0] + float(lin[1] @ q_bits)

    def __len__(self) -> int:
        return self._k
