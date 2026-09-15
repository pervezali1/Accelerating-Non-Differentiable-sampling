# The same experiments without the constraint

Companion to `Constrained_Sampling_ANDS_ball_constrained.ipynb`. Same field
$J_s(x)=s\,[x]_\times$, same two targets, same $J=0$ against $s=8,16$, same $W_1$-to-floor
methodology, same 8 replicas of 5000 walkers. The ball $K=\{\lVert x\rVert\le0.5\}$ is removed
and the projection drops out of the update.

**Summary in one line:** on the ball the field was free in accuracy and bought 1.5–1.9×; without
the ball it is not free, it is not stable, and at $s=16$ it does not have a stationary
distribution at all.

---

## 0. Two settings that had to change, and why

**`MAXIT` 5000 → 80000.** At the original horizon the unconstrained reversible chain has not
converged — it is at 12.9× the sampling floor on Target A and 8.8× on Target B. Keeping 5000
would have measured the transient, not the stationary law.

| iterations | A: $W_1$ | × floor | B: $W_1$ | × floor |
|---|---|---|---|---|
| 5 000 | 0.755 | 12.90 | 0.402 | 8.84 |
| 10 000 | 0.540 | 9.22 | 0.246 | 5.42 |
| 20 000 | 0.337 | 5.77 | 0.121 | 2.65 |
| 40 000 | 0.164 | 2.80 | 0.066 | 1.46 |
| 60 000 | 0.127 | 2.18 | 0.052 | 1.14 |

For scale, $\tau$ to 2× floor for $J=0$ is **52 950** on A and **24 050** on B, against 381 and
469 on the ball. Truncating at $R=0.5$ was not a mild restriction — it removed roughly two orders
of magnitude of the mixing problem.

**Anisotropy, reported consistently.** The constrained summary compared "anisotropy 1.18 and
1.32" against "34.6 unconstrained", but 34.6 is $\mathrm{cond}(\Sigma)$ while 1.18 is a ratio of
coordinate standard deviations. Like for like: the sd anisotropy is **4.00** and **4.11** here
against 1.18 and 1.32 on the ball, and $\mathrm{cond}(\Sigma)=34.6$ in *both* settings, since
truncation does not change $\Sigma$. The contrast is real and about 3×, not 30×.

Exact target: coordinate sds $(1.01,2.02,4.05)$ on A and $(0.69,1.42,2.84)$ on B; mean
$\lVert x\rVert$ 3.736 and 2.605; $q_{99}\lVert x\rVert$ about 26× the old ball radius.

---

## 1. Experiment 3 — head to head

Stationary mean $W_1$ over the last 10 tracking points, 8 replicas, and the coordinate standard
deviations as a ratio to the exact target (1.000 = correct).

| target | scheme | stationary $W_1$ | × floor | mean $\lVert x\rVert$ | sd ratio to exact |
|---|---|---|---|---|---|
| A | $J=0$ | 0.0771 | **1.3×** | 3.612 | [0.97, 0.98, 0.95] |
| A | $s=8$ | 0.4493 | 7.7× | 4.714 | [1.33, 1.38, 1.29] |
| A | $s=16$ | 45.57 | 779× | 106.0 | [42.9, 58.8, 20.7] |
| B | $J=0$ | 0.0433 | **1.0×** | 2.615 | [1.02, 1.00, 0.99] |
| B | $s=8$ | 12.18 | 268× | 30.68 | [35.3, 39.6, 21.1] |
| B | $s=16$ | $1.28\times10^{8}$ | $2.8\times10^{9}$× | $4.0\times10^{8}$ | [2.4e8, 2.8e8, 1.2e8] |

Exact means are 3.736 and 2.605.

**The reversible scheme sits essentially on the sampling floor** — 1.3× and 1.0× — and
reproduces every coordinate sd to within 5%. That is the correct answer and it is what the
unconstrained problem looks like when it works.

**No $J_s$ run has a measurable $\tau$.** Not because they are slow: their stationary error never
comes near the threshold, so there is nothing to time. A speed-up cannot be quoted for a chain
that is not converging.

**$s=16$ on Target B misses the coordinate scales by eight orders of magnitude.** In the 2-d
marginal cell, the fraction of walkers still inside the plotted window is 17.6% for A at $s=16$
and **0.0%** for B — not one of 40 000 walkers remains in frame.

---

## 2. Experiment 4 — where the boundary atom went

On the ball the discretisation artefact was a mass atom on $\partial K$: flat in $s$ (as it had
to be, $J_s$ being exactly tangential) and scaling like $\sqrt\eta$. There is no wall here, so
the artefact surfaces instead as **radial inflation**, and unlike the atom it is *not* flat in
$s$.

Measured by starting every walker at an exact draw from $\pi$, so there is no transient to
subtract; 40 000 steps at each setting. Entry is $E\lVert x\rVert/E\lVert x\rVert_\pi-1$.

| $\eta$ | A: $s=0$ | $s=8$ | $s=16$ | B: $s=0$ | $s=8$ | $s=16$ |
|---|---|---|---|---|---|---|
| $5\times10^{-4}$ | −1.60% | +23.99% | +596.9% | +0.77% | +189.4% | **+1 725 943%** |
| $2.5\times10^{-4}$ | −1.95% | +4.93% | +41.0% | +0.45% | +21.0% | +539.0% |
| $1.25\times10^{-4}$ | −1.93% | +0.61% | +7.28% | +0.33% | +6.92% | +32.4% |

**The reversible baseline is flat in $\eta$** (−1.9% on A, +0.3…0.8% on B), so everything above
it is the field.

**The $s^2$ law holds where the scheme is stable.** At the finest stepsize the field's
contribution (after subtracting the $s=0$ row) is 2.54% at $s=8$ and 9.21% at $s=16$ on A — a
ratio of **3.63** — and 6.59% / 32.08% on B — a ratio of **4.87**. Both bracket the predicted
$(16/8)^2=4$.

**The much steeper ratios at coarse $\eta$ are the runaway regime, not a failure of the law.**
Going $\eta\to\eta/2$ shrinks the $s=16$ contribution by 13.9× on A, far more than the
perturbative 2×, because at $\eta$ the chain is past the escape radius below and escaping rather
than merely biased.

---

## 3. Experiment 4b — the escape radius

At large $\lVert x\rVert$, per step,

$$\underbrace{-2\eta\,x\!\cdot\!g}_{\sim-2\eta c\lVert x\rVert}
\ +\ \underbrace{\eta^2s^2\lVert x\times g\rVert^2}_{\sim+\eta^2s^2c^2\lVert x\rVert^2},\qquad c=\lVert g\rVert .$$

The skew term contributes nothing at first order — it is exactly tangential — so this is entirely
the $\lVert a\rVert^2$ in $\lVert x\rVert^2\mapsto\lVert x\rVert^2+\lVert a\rVert^2$. They balance at

$$r^\*=\frac{2}{\eta\,s^2\,c},$$

and past $r^\*$ the outward term wins, so a walker there is pushed further out, which strengthens
the push. **Prediction: the chain escapes once $r^\*$ falls below the radius the target reaches.**

The two targets differ in $c$ and it decides which breaks first: the elliptical gradient
saturates at a *unit* vector through $M^{-1}$ ($c\le2.76$), the $\ell_1$ gradient at a *sign*
vector, which is longer ($c\le4.62$).

$\max\lVert x\rVert$ after 80 000 iterations, 5000 walkers:

| $s$ | $r^\*$ (A) | A (reach ≈ 22) | $r^\*$ (B) | B (reach ≈ 15) |
|---|---|---|---|---|
| 0 | ∞ | 21.9 ✓ | ∞ | 15.4 ✓ |
| 2 | 362 | 21.0 ✓ | 216 | 15.6 ✓ |
| 4 | 90 | 20.8 ✓ | 54 | 17.9 ✓ |
| 8 | 22.6 | 39.5 ✗ | 13.5 | **1 165** ✗ |
| 16 | 5.7 | **674** ✗ | 3.4 | $3.2\times10^{9}$ ✗ |

Every outcome matches the prediction, including the ordering between targets. **$s=4$ is the
largest usable strength at this stepsize**, on both targets.

Since $r^\*\propto1/\eta$, refining the stepsize buys *validity* here, not just accuracy, and it
does:

| target | $\eta$ | $r^\*$ | $\max\lVert x\rVert$ @80k | exact reach |
|---|---|---|---|---|
| A | $5\times10^{-4}$ | 5.7 | 674 | 22 |
| A | $2.5\times10^{-4}$ | 11.3 | 38.1 | 22 |
| A | $1.25\times10^{-4}$ | 22.6 | **17.0** ✓ | 22 |
| B | $5\times10^{-4}$ | 5.7 | $3.2\times10^{9}$ | 15 |
| B | $2.5\times10^{-4}$ | 11.3 | 1 098 | 15 |
| B | $1.25\times10^{-4}$ | 22.6 | 24.1 | 15 |

**This reframes the constrained result.** $R=0.5$ sits far inside every $r^\*$ in these tables.
The projection was not merely absorbing the artefact into the boundary atom; it was holding
walkers inside the radius where the scheme is stable at all. "The field is free in accuracy" was
a true statement about that experiment and not a property of the field.

---

## 4. Experiment 5 — the stepsize trade

On the ball the currency was the boundary atom, flat in $s$, so the speed-up was pure profit and
the trade was arithmetic. Here the currency grows with $s$, and the question is not whether the
trade pays but whether refinement can make $s=16$ usable at all. It cannot.

$\tau$ is to 3× the floor; "never" means the chain does not reach it within the run.

| target | scheme | $\eta$ | $\tau(3\times)$ | stationary $W_1$ | × floor |
|---|---|---|---|---|---|
| A | $J=0$ | $5\times10^{-4}$ | **38 300** | 0.0771 | 1.3× |
| A | $s=16$ | $5\times10^{-4}$ | never | 45.57 | 779× |
| A | $s=16$ | $2.5\times10^{-4}$ | never | 2.907 | 49.7× |
| A | $s=16$ | $1.25\times10^{-4}$ | never | 0.469 | 8.0× |
| B | $J=0$ | $5\times10^{-4}$ | **18 250** | 0.0433 | 1.0× |
| B | $s=16$ | $5\times10^{-4}$ | never | $1.28\times10^{8}$ | $2.8\times10^{9}$× |
| B | $s=16$ | $2.5\times10^{-4}$ | never | 5220 | 115 000× |
| B | $s=16$ | $1.25\times10^{-4}$ | never | 11.98 | 264× |

Refining by 4× and paying 4× the iterations improves the stationary error by 97× on A and
$10^{7}$× on B, and still leaves $s=16$ at 8× and 264× the floor with no measurable $\tau$.

(The radial inflations quoted in this experiment — +27.2% on A and +1060% on B at $\eta/4$ — are
larger than Experiment 4's at the same $\eta$ because these runs are 320 000 steps from a point
start while Experiment 4 is a fixed 40 000 steps from stationarity. Different quantities; the
ordering between them is what it should be.)

---

## 5. What to take from this

1. **Unconstrained, use $J=0$ at these settings.** It sits on the sampling floor on both targets.
   Every $J_s$ run at $s\ge8$ is wrong, and no speed-up is quotable because none of them converge.
2. **If you want the field unconstrained, respect $r^\*=2/(\eta s^2c)$.** Keep $r^\*$ comfortably
   above the radius your target actually reaches. Here that means $s\le4$ at
   $\eta=5\times10^{-4}$, and $s$ can grow only as $\eta^{-1/2}$.
3. **Measure $c$ on your own target.** It is the $\ell_1$ target's longer saturated gradient that
   makes it fail at half the $s$ the elliptical one tolerates. A bound from
   $\lVert M^{-1}\rVert$ alone would have missed that.
4. **The ball was doing more than bounding the domain.** Every accuracy conclusion in the
   constrained study is about the projected scheme inside a region where the field is harmless.
   That does not transfer.

## Caveats

- $\tau$ and stationary $W_1$ for divergent runs are reported as measured; they describe how far
  the chain has gone, not a stationary law, because there is no stationary law to describe.
- Experiment 4 uses 4 replicas rather than 8 (moments converge far faster in $N$ than $W_1$ does).
  Its one marginal claim — that $s=8$ at $\eta/4$ has slightly *lower* stationary $W_1$ than
  $J=0$, 0.0643 against 0.0707 — is inside plausible Monte Carlo noise and should not be leaned on.
- The stability probes use 5000 walkers and one chain per setting. They are tail statistics and
  the effects are orders of magnitude, but they carry no error bars.
- Everything is at $\delta=0.1$, $d=3$, and this one $\Sigma$.
