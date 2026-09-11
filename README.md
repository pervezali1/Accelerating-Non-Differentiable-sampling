# Accelerating Non-Differentiable Sampling

Anchored Langevin sampling for potentials that are only ever **evaluated**, never
differentiated, with an antisymmetric (skew) drift field added for circulation — and a hard
convex constraint handled by projection.

This repository carries the method from the synthetic 3-dimensional study in
`Constrained_Sampling_ANDS_ball_constrained.ipynb` over to two real binary-classification
posteriors:

| dataset | source | n | d | constraint radius R |
|---|---|---|---|---|
| **Titanic** | Kaggle training split (891 passengers) | 891 | 10 | 1.4 |
| **MAGIC Gamma Telescope** | UCI `magic04` (stratified 2 000-row subsample of 19 020) | 2 000 | 11 | 1.9 |

The main artefact is [`notebooks/Constrained_Sampling_ANDS_real_data.ipynb`](notebooks/Constrained_Sampling_ANDS_real_data.ipynb).

## The problem

For labels $y_i \in \{-1,+1\}$ and standardised features $\phi_i$, write $\psi_i = y_i \phi_i$.
The Gibbs posterior over classifier weights is

$$U(w) = \tau\,\frac{1}{n}\sum_i \max(0,\,1 - w^\top\psi_i) \;+\; \lambda\lVert w\rVert_1,
\qquad \pi(w) \propto e^{-U(w)}\mathbf 1_K(w),\qquad K = \{\lVert w\rVert_2 \le R\}.$$

Both terms are non-differentiable — the hinge kinks at every active margin, the $\ell_1$ prior on
every coordinate axis — so the sampler works from a smoothed *anchor* $U_0 \ge U$ and the time
change $e^{\Delta}$, $\Delta = U - U_0$. The hinge term is an **average**, not a sum, which keeps
the anchor gap $O(\tau\delta)$ instead of $O(n\tau\delta)$; with a sum, $e^{\Delta}$ underflows on
any real dataset.

## What the real-data setting adds

The dynamics for a state-dependent skew field is

$$dX_t = e^{\Delta(X_t)}\Bigl[-\bigl(I + J(X_t)\bigr)\nabla U_0(X_t) + \nabla\!\cdot J(X_t)\Bigr]dt
+ \sqrt{2}\,e^{\Delta(X_t)/2}\,dW_t,$$

and projection onto $K$ imposes a second condition, $J(x)\nu(x) = 0$ on $\partial K$.

The 3-dimensional notebook used the axial field $J_s(x)w = s\,(x \times w)$, which is a cross
product — it exists only in $d = 3$, and it is divergence-free, so the correction term
$\nabla\!\cdot J$ was derived but never needed. Its $d$-dimensional replacement, for a fixed
antisymmetric $A$,

$$J_s(x) = \frac{s}{R^2}\Bigl[\lVert x\rVert^2 A + x\,(Ax)^\top - (Ax)\,x^\top\Bigr],
\qquad \nabla\!\cdot J_s(x) = -\frac{s}{R^2}(d-2)\,Ax,$$

still annihilates $x$ identically (so $J_s\nu = 0$ on the wall by construction) but is **not**
divergence-free once $d > 2$. At $d = 10$ and $d = 11$ the correction is the same order as the
drift, so the notebook can run the ablation the synthetic study could not.

## What the runs show

Scored against two independent exact random-walk Metropolis references (acceptance 0.240 and
0.234), with the $W_1$ floor — the level at which two samples of this size are
indistinguishable — measured rather than assumed.

**Head-to-head at skew strength $s = 4$.** The constant field is antisymmetric and
divergence-free; it fails only $J\nu = 0$, and that one defect costs a factor of six.

| | mean $W_1$ | max KS | mass near $\partial K$ |
|---|---|---|---|
| Titanic — $J = 0$ | 0.0061 | 0.037 | 6.80% |
| Titanic — constant $J_a$ | **0.0387** | **0.218** | **16.50%** |
| Titanic — state-dependent $J_s$ | 0.0065 | 0.038 | 7.20% |
| Titanic — reference (truth) | 0.0063 | — | 0.56% |
| MAGIC — $J = 0$ | 0.0096 | 0.041 | 3.75% |
| MAGIC — constant $J_a$ | **0.0409** | **0.193** | **10.30%** |
| MAGIC — state-dependent $J_s$ | 0.0092 | 0.045 | 3.50% |
| MAGIC — reference (truth) | 0.0084 | — | 0.32% |

**Dropping $\nabla\!\cdot J$ is worse than using an inadmissible field** — the experiment the
3-dimensional study could not run, since there the correction was identically zero.

| mean $W_1$ | $s=0$ | $s=1$ | $s=2$ | $s=4$ | $s=8$ |
|---|---|---|---|---|---|
| Titanic — $J_s$ with the correction | 0.0061 | 0.0059 | 0.0055 | 0.0065 | 0.0131 |
| Titanic — $J_s$, correction dropped | 0.0061 | 0.0284 | 0.0490 | 0.0724 | **0.1014** |
| MAGIC — $J_s$ with the correction | 0.0096 | 0.0097 | 0.0090 | 0.0092 | 0.0140 |
| MAGIC — $J_s$, correction dropped | 0.0096 | 0.0353 | 0.0623 | 0.1047 | **0.1534** |

The two failures leave different fingerprints. The uncorrected field *is* tangential, so its
near-boundary mass barely moves (6.80% → 10.20% across the sweep on Titanic, against
6.80% → 27.15% for the constant field); its damage is interior, where the flux it generates is
no longer divergence-free. Boundary mass diagnoses a violated wall condition and is blind to a
missing correction term.

**Accuracy sees none of it.** Across the whole sweep — mean $W_1$ ranging over a factor of 16 —
every method's classification accuracy stays inside the reference posterior's own 5-95% spread.
On MAGIC the per-draw accuracy actually *rises* with the bias (0.7832 → 0.7855 for the constant
field, 0.7832 → 0.7859 for the uncorrected one, as $s$ goes 0 → 8) while the admissible field
sits on the reference at 0.7831: the two most badly biased samplers score highest.
$\operatorname{sign}(w^\top\psi_i)$ is invariant under $w \mapsto cw$, so a scale-invariant
statistic is structurally blind to the radial distortion these fields produce.

| | catches a violated $J\nu = 0$ | catches a missing $\nabla\!\cdot J$ |
|---|---|---|
| mass near $\partial K$ | yes | **no** |
| accuracy | **no** | **no** |
| $W_1$ against a trusted reference | yes | yes |

**The learning curve** (`figures/learning_curve_*.png`) is the accuracy plot that does carry
information. Every chain starts at $w = 0$, where every margin is exactly zero and accuracy is
exactly chance, so the trace runs 0.5 up to the posterior's own rate — 0.789 on Titanic, 0.783 on
MAGIC, reached to within 1% inside ~100-200 iterations. That ceiling is not 1: these are linear
classifiers on real, noisy data, and a sampler that climbed past its own target's rate would be
reporting a bug. During the climb the inadmissible constant field is the slowest of the three
(0.711 against 0.749 at iteration 40 on MAGIC).

These accuracy numbers are **in-sample** — the posterior is fitted to all $n$ rows, and the
question asked is whether sampling bias reaches a decision statistic, not how well the model
generalises.

**Refining the stepsize fixes the projection atom and cannot fix a wrong boundary condition.**
At fixed physical time $T = 0.6$ on Titanic, the $J = 0$ atom falls 6.80% → 4.50% → 3.10% as
$\eta$ is refined 4×, with atom$/\sqrt\eta$ nearly constant (3.93, 3.67, 3.58), while the
constant-$J$ $W_1$ plateaus at 0.0387 → 0.0371 → 0.0397.

## Layout

```
ands/
  targets.py       the hinge + L1 Gibbs posterior, its anchor and projection
  penalties.py     Lasso, MCP, SCAD with exact Gaussian smoothing and anchor bounds
  data.py          Titanic, MAGIC Gamma Telescope, Wisconsin breast cancer
  skew.py          the three fields, closed-form divergences, autograd checks
  samplers.py      projected anchored Langevin; exact random-walk Metropolis reference
  diagnostics.py   W1 per coordinate, KS, boundary mass, reference floor
  experiments.py   the experiment grid and its result cache
  plots.py         figures
notebooks/         the executed study
tests/             what the theory claims, asserted
```

## The ball-constrained study

[`notebooks/Constrained_Sampling_ANDS_ball_constrained.ipynb`](notebooks/Constrained_Sampling_ANDS_ball_constrained.ipynb)
compares $J = 0$ against the admissible state-dependent skew

$$J_s(x) = \frac{s}{R^2}\Bigl[\lVert x\rVert^2 A + x(Ax)^\top - (Ax)x^\top\Bigr]$$

on the ball $K = \{\lVert x\rVert_2 \le R\}$, at $s = 4, 8, 16$. The constant $J_a$ does not appear: on a
constrained problem it is not a candidate, since it fails $J\nu = 0$ on $\partial K$ by construction.

$J_s$ annihilates $x$ identically, so tangency at the wall is structural rather than tuned — it holds to
machine precision at every strength. The price is $\nabla\!\cdot J_s = -\tfrac{s}{R^2}(d-2)Ax \neq 0$,
carried in the drift and checked against autograd.

| | $\tau$ | speed-up | stationary $W_1$ | mass on $\partial K$ |
|---|---|---|---|---|
| **A** $J = 0$ | 3831 [3634, 4075] | — | 0.0137 [0.0122, 0.0155] | 2.62% |
| A $J_s$, $s=8$ | 2191 [2109, 2275] | 1.75× [1.64, 1.88] | 0.0136 [0.0119, 0.0155] | 2.62% |
| A $J_s$, $s=16$ | 1809 [1600, 2106] | **2.12× [1.79, 2.44]** | 0.0141 [0.0126, 0.0159] | 2.58% |
| **B** $J = 0$ | 4200 [4069, 4341] | — | 0.0116 [0.0104, 0.0128] | 2.09% |
| B $J_s$, $s=8$ | 2806 [2662, 2956] | 1.50× [1.41, 1.59] | 0.0125 [0.0111, 0.0140] | 2.08% |
| B $J_s$, $s=16$ | 1494 [1447, 1550] | **2.81× [2.68, 2.94]** | 0.0130 [0.0110, 0.0152] | 2.10% |

At these strengths the acceleration is **free**: every stationary $W_1$ interval overlaps $J = 0$'s and
straddles the sampling floor, and the projection atom on $\partial K$ is flat to within 0.07 percentage
points across all four schemes.

**Getting the atom below $J = 0$.** At fixed $\eta$ the projection atom is flat across $s$, and must be:
it is a discretisation artefact driven by the *normal* step near the wall, and $J_s$ is tangential. But it
scales like $\sqrt\eta$, so the speed-up can be spent on refining $\eta$. At $s = 16$ with $\eta/2$,
$J_s$ reaches the floor in **1.24× and 1.38× fewer iterations** than $J = 0$ *and* leaves **31% less mass**
on the wall (1.82% against 2.62%, 1.44% against 2.09%). Refining to $\eta/4$ drops the atom further but
costs more iterations than the speed-up affords.

Aiming is most of the effect. At the same $s = 16$, rotating in the plane of the two largest-variance
eigenvectors gives 2.12× and 2.81×; the two smallest gives 1.04× [0.96, 1.13] and 1.14× [1.08, 1.20] —
nothing. That is the argument for this field over the cross product $s\,(x \times w)$, which is equally
admissible but rotates about $x$, so its plane is set by position rather than by $\Sigma$.

All numbers are 8 replicas with bootstrap intervals at a horizon where $J = 0$ reaches the floor
(8000 iterations).

## The unconstrained companion study

[`notebooks/anchored_langevin_unconstrained_J.ipynb`](notebooks/anchored_langevin_unconstrained_J.ipynb)
runs the same $J=0$ against constant-$J$ comparison with the constraint removed, on Lasso, MCP and SCAD
with the closed-form Gaussian smoothing of a piecewise-quadratic penalty. Removing $K$ removes $\psi$:
invariance is free for a constant skew, but the naive carry-over $\nabla\psi=-x$ gives a drift
$\propto e^{U}x$ that overflows within 400 steps off a compact set, and the tempered choice
$\psi=-e^{-U_0}$ collapses the update back to anchored non-reversible Langevin.

$\alpha$ is held fixed throughout — no annealing schedule — so the skew drift $\alpha J$ is constant in
time as well as in space, and the strength has to be chosen outright. Findings from re-running with twelve
replicas instead of three, and sweeping $\alpha$ over $\{1,2,4,8\}$:

- **The best constant $\alpha$ is 8, not 2**, and at that strength the $\tau(0.06)$ speed-up is
  2.73× [2.58, 2.89] on Lasso, 3.01× [2.82, 3.22] on MCP and 2.82× [2.69, 2.94] on SCAD. This
  **overturns** the conclusion that removing the constraint reduces what the skew buys: that claim
  (1.4–2.0× unconstrained against 2.8–3.1× on the ball) came from stopping the sweep at $\alpha=4$.
  A properly tuned constant skew gets the constrained figure. Axis alignment matters more as $\alpha$
  grows — at $\alpha=8$ the tridiagonal axis gives 1.26× where the aligned one gives 2.90×.
- **At $\alpha=8$ the stationary cost is real**, and it is a measured trade-off rather than a worry: on
  SCAD the intervals do not overlap (0.0234 [0.0215, 0.0252] against 0.0309 [0.0281, 0.0335]). At
  $\alpha=2$ there is no measurable cost at all. Roughly 2.8× in mixing for roughly 30% in stationary
  $W_1$.
- **$a\le1$ is false for MCP and SCAD.** That bound is Jensen and needs convexity; both penalties are
  non-convex by construction. The replacement is proven from the heat-semigroup representation —
  $\max_t(p-p_0)\le\mu^2\max(0,-c_2^{\min})$, giving $a\le1.046$ and $1.070$ — and is attained, not
  merely valid. The tempering argument survives, but because $g-g_0$ is bounded, not because the anchor
  dominates.

The smoothing lemma and the bound live in [`ands/penalties.py`](ands/penalties.py), with
`tests/test_penalties.py` pinning both.

## The multivariate Laplace study

[`notebooks/Unconstrained_Sampling_ANDS.ipynb`](notebooks/Unconstrained_Sampling_ANDS.ipynb) compares
$J = 0$ against a constant skew on two multivariate Laplace targets. Unconstrained, every constant
antisymmetric $J$ leaves $\pi$ invariant, so the choice is purely a rate choice — and the original
tridiagonal $J_a$ made it arbitrarily.

| at $a = 4$ | Target A | Target B |
|---|---|---|
| $J_a$ tridiagonal | 1.66× [1.59, 1.74] | 1.58× [1.46, 1.72] |
| $J$ **aligned** to the 2 largest-variance eigenvectors | **3.18× [3.02, 3.34]** | **3.41× [3.12, 3.73]** |
| $J$ anti-aligned (2 smallest) | 1.10× [1.03, 1.19] | 1.11× [1.04, 1.18] |

Aiming is worth about a factor of two on top of the strength, and it is what lets the strength be raised:
the aligned field improves to 3.69× and 4.56× at $a = 8$, where the tridiagonal one stalls on A and becomes
unmeasurable on B. At $a \le 4$ the acceleration is free — stationary $W_1$ sits at 0.83-0.99× the floor.

The notebook also carries a correction to the measurement: estimating the $W_1$ floor against one fixed
reference has a 15-20% per-coordinate spread, which on Target B put the $\tau$ threshold inside the
stationary noise and made every speed-up look negative. Drawing both sides fresh over 24 pairs, and
setting the threshold at $3\times$ the floor, fixes it; a guard labels any row where $\tau$ is still not
resolvable.

## Standalone, in one file

[`examples/learning_curve_standalone.ipynb`](examples/learning_curve_standalone.ipynb) asks whether the
skew field speeds up the *accuracy* learning curve, with NumPy, pandas and matplotlib only — no torch and
nothing from the `ands` package. The same thing as a command-line script is
[`learning_curve_standalone.py`](examples/learning_curve_standalone.py).

**$J_s$ is a real speed-up; $J_a$ is not.** Iterations to 99% of the exact posterior's accuracy, 6
replicas with bootstrap intervals:

| | $J_s$, $s=8$ | $J_s$, $s=16$ | $J_a$, $s=8$ |
|---|---|---|---|
| Titanic | **1.51× [1.45, 1.56]** | **1.77× [1.70, 1.84]** | 0.85× [0.83, 0.88] |
| MAGIC | **1.33× [1.29, 1.38]** | **1.43× [1.37, 1.49]** | *2.57×, but overshoots* |

At $s \le 8$ the $J_s$ gain is free — it lands on the ceiling to within 0.0007. $J_a$ is slower than
$J = 0$ on Titanic, and on MAGIC its apparent 1.8-4× is an artefact: its final accuracy sits *above* the
exact posterior's, so it crosses the threshold on its way past. That is the constrained study's bias
showing up as speed.

Two of the three fixes were about measurement: the strength had never been swept ($s = 4$ was the weakest
setting on the grid), and accuracy saturates near 0.79 within ~100 iterations so a linear axis buries every
difference in the top 2% of the panel. Plotting $\text{ceiling} - \text{accuracy}$ on a log axis separates
the same runs by two orders of magnitude.

The notebook also records a negative result: aiming the rotation at the posterior's two largest-variance
directions, worth 2-3× on $W_1$, is worth **nothing** here (0.98× and 1.00×) and loses to an arbitrary
tridiagonal generator. $W_1$ is dominated by the large-variance directions; accuracy reads only the
direction of $w$, so a generator that couples all coordinates serves it better.

## Does the regularizer change the answer?

[`notebooks/Regularizer_comparison.ipynb`](notebooks/Regularizer_comparison.ipynb) runs all three
penalties in `penalties.py` — Lasso, MCP, SCAD — against all five schemes on all three real datasets,
and asks for which combination a skew field actually buys accuracy over $J = 0$.

**It does not. The field does.** The exact posterior's accuracy ceiling is the same whichever penalty
is used — Titanic 0.7898/0.7898/0.7901, MAGIC 0.7837/0.7833/0.7836, breast cancer
0.9599/0.9605/0.9593 — every spread below the 0.002 tolerance used to call two accuracies
indistinguishable. That is not a badly-chosen $\lambda$: at $\lambda = 0.25$, $a = 3$ the coefficients
do reach into MCP's and SCAD's taper, where the three charge 0.375 / 0.094 / 0.125 at $|w| = 1.5$. The
penalties genuinely differ; the classifier does not.

The accuracy curves themselves are in
[`notebooks/figures/reg_accuracy_curves.pdf`](notebooks/figures/reg_accuracy_curves.pdf): accuracy
rising from 0.5 toward the exact posterior, with the elbow zoomed underneath and a dot marking where
each scheme first reaches 99% of the ceiling. Both $J_s$ curves sit left of $J = 0$ in all three
panels. On MAGIC the dashed $J_a$ curve visibly climbs *above* the ceiling line, which is the
overshoot in its plainest form. On Titanic $J = 0$ overtakes $J_s$ after about 250 iterations, a
reminder that reaching the ceiling sooner and settling on it exactly are different things.

Counting wins, where a scheme wins only if its speed-up interval clears 1 **and** its gap to the ceiling
is no worse than $J = 0$'s by more than 0.002:

| field | wins |
|---|---|
| $J_s$, tangent to $\partial K$ | **15** of 18 |
| $J_a$, constant | **0** of 18 |

split evenly across the penalties — Lasso 5, MCP 5, SCAD 5. $J_s$ at $s = 8$ wins in all nine
dataset-by-penalty cells and is the only setting that never trades accuracy for its speed-up
(1.53-1.55× on Titanic, 1.33-1.35× on MAGIC, 1.11-1.15× on breast cancer).

$J_a$ fails in both available ways. On Titanic and breast cancer it is simply slower, 0.23-0.66×: a
constant skew is not tangent to the sphere, so it drives walkers into the wall and projection discards
the work. On MAGIC it looks like 2.1-3.3× but its gap to the ceiling is *negative*, so those chains
settle **above** the exact posterior's own accuracy — bias flattering itself on the one metric that
cannot see bias, which is the constrained study's finding again.

This notebook is also where $a > 1$ first appears on real data: breast cancer under MCP (1.0056) and
SCAD (1.0043), the non-convexity correction doing visible work, inside the proven bound both times.
All accuracies are in-sample.

## Running it

```bash
pip install numpy scipy pandas matplotlib seaborn torch jupyter pytest
pytest tests/ -q
jupyter nbconvert --execute --inplace notebooks/Constrained_Sampling_ANDS_real_data.ipynb
```

Results are cached in `results/` keyed by a hash of their settings, so re-execution is cheap and
changing a setting invalidates only what it touches. Raw data is cached in `data/`; the loaders
fetch it once from public mirrors.
