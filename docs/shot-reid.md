# Shot re-identification: design notes

How `shotmem.py` + `reid.py` decide that a shot is a recurrence of one already seen.
The README has the summary; this is the reasoning, the constants, and what is known to
be wrong with it.

## The decision

At each cut the new shot's first frame `q` (a 64-bit luma pHash) is matched against
memory. Retrieval is a framestore Hamming query within `reid_maxd`; everything after is
scoring. The rule is one posterior over *which group, or a new one*, under a
Chinese-restaurant prior:

```
P(g   | q)  ~  n_g * sum_k w_gk P(q | prototype k of g)
P(new | q)  ~  alpha * P(q | population)
```

re-ID iff the leader's posterior log-odds clear `reid_llr` nats. Written out, with
`ll_k = log P(q | k)`:

```
logit_g = log n_g + logsumexp_k(log w_gk + ll_k)
odds    = logit_g* - logsumexp( {logit_h : h != g*} u {log alpha + log P(q|population)} )
```

Every term earns its place:

| term | fixes |
|---|---|
| `P(q\|population)` | an absolute likelihood is not comparable across profiles of different entropy; and bits every setup in the footage shares must stop carrying evidence |
| mixture over `k`, not max | max gives a group one independent attempt at the threshold per prototype, so a group spread over a pan out-competes a tight one |
| `w_gk` = opening share | the query is a shot-*opening* frame; frame mass answers which prototype a *random* frame resembles |
| `n_g` | a setup that has recurred five times really is likelier to recur than a one-off |
| rival `logit_h` in the denominator | best-vs-second-best margin, and it scales the bar with the number of candidates |

**Why log-odds and not a probability.** `ll` is a naive-Bayes sum over pHash bits, which
are strongly correlated (low-frequency DCT coefficients of natural images), so its
effective degrees of freedom are well below 64 and its magnitude is overconfident. A
threshold reading `P >= 0.9` would not mean what it says. Calibrating it properly needs
labelled data. Log-odds keep the threshold in nats, and — by construction — a lone
candidate holding one shot at `alpha = 1` scores *exactly* its bare likelihood ratio, so
the prior and the competition only act when there is something to weigh against.

## Everything is linear in the query bits

```
log P(q | p) = sum_i log(1-p_i) + sum_i q_i log(p_i/(1-p_i)) = const + logit . q
```

So a profile is a scalar and a 64-vector of log-odds weights, computed once per shot at
`finalize()`; subtracting a null is subtracting its `const` and `logit`. The whole thing
is a linear classifier over hash bits whose weights are learned per shot. Scoring a
candidate is one dot product, and the query is unpacked exactly once per cut.

## Constants

| constant | default | what it is |
|---|---|---|
| `reid_maxd` | 0.25 | recall radius, relative Hamming. Loose on purpose — precision is the scorer's job. |
| `reid_llr` | 8.0 | accept threshold, nats. An agreeing distinctive bit is worth ~0.7 and a disagreeing one costs ~3.9, so this asks for roughly a dozen distinctive bits in agreement. |
| `reid_eps` | 0.02 | per-bit flip rate the model always allows. |
| `reid_alpha` | 1.0 | CRP concentration: prior weight on "a new setup". 1.0 makes a lone candidate score its bare ratio. |
| `OPEN_PRIOR` | 1.0 | pseudocount on the mixture weights (`shotmem.py`). |

`reid_eps` is **not** a noise parameter to fit. Fitting it to within-shot
distance-to-prototype measures the wrong channel: within-shot variation is already
modelled by the Bernoulli `p`, and the gap that matters — how a frame of occurrence 1
differs from a frame of occurrence 2 — is by construction never observed inside a shot.
Estimating it from accepted re-IDs is selection-truncated and biases itself downward in a
loop. Treat it as a contamination floor asserting no bit is ever certain, the role
variance flooring plays in GMM-UBM.

`reid_llr` was swept on synthetic memories of 40 groups. 8 is where the accept radius
comes out roughly content-independent (~10 bits both when all 64 bits are usable and when
44 are shared), which is what makes a fixed nat threshold usable at all; 16 is unusable on
shared-layout footage, where there is simply less total evidence available.

## Timing edges

The cut is confirmed one frame late, so the frame at `t-1` is the new shot's first.
Two consequences, both deliberate:

* **Accumulation lags one frame.** A frame folds into a group's profile only once the
  next frame has said which shot owns it. Otherwise every cut folds the new shot's first
  frame — the one frame guaranteed to show a different setup — into the outgoing group,
  which was measurably the largest single source of error before it was fixed.
* **One frame is labelled late.** The frame that opens a shot is emitted with the
  outgoing `shot_id`. Fixing that means holding emission back a frame, which defeats the
  point of deciding at the first frame.

**Call `ShotTracker.close()` (or `Publisher.close()`) at end of stream.** Nothing else can
know the last shot is over, so without it the final shot never reaches `shots`, its frames
never fold into its group's profile, and its prototypes never vote in the null.

## Known limitations

* **Groups never merge.** If one setup gets split across two groups, the margin term now
  makes them *block each other*, and a third occurrence opens a third group rather than
  joining either. That is the safe behaviour for a system that cannot merge, but it means
  over-splitting can compound. A blocked match is precisely the signal that two groups
  should be merged; making prototypes the only unit and groups connected components over
  them (union-find) is the fix.
* **`shot_group_id` is immutable once emitted.** The frame-1 decision is a prediction
  made from a single, often motion-blurred, post-cut frame. The whole shot's counts would
  be a far better estimate, but using them means deciding at shot close, which defeats the
  purpose (recognising a setup at frame 1 to reuse its scene graph instead of re-running
  heavy models).
* **The evidence scale is content-dependent.** Footage where only 20 bits are distinctive
  has less total evidence available than footage where all 64 are, and no threshold choice
  fixes that — only more bits do.
* **One 64-bit luma pHash.** `imfeat` already computes aHash/wHash/pHash per channel and
  eight of the nine go unused. The scorer is now safe to widen: a bit unstable within
  shots is muted by `p -> 1/2`, and a bit uninformative across shots is muted by the
  population term, so bad bits self-mute rather than drowning the score.
* **Adjacent shots are not cannot-linked.** Consecutive shots are different setups by
  definition, so forbidding `group(shot_i) == group(shot_i-1)` is free precision that is
  not yet taken.
