# Accelerating-Non-Differentiable-sampling

## Anchored Langevin on a univariate Laplace target

`anchored_langevin_laplace_1d.ipynb` runs the protocol of `beta_nonrever_Gaunssian.ipynb`
(N-particle ensemble from a common start, W1 to reference samples recorded every iteration,
density plots of the terminal ensemble) with **anchored Langevin** on the univariate Laplace
target `pi(x) ∝ exp(-|x-mu|/b)`, which is non-differentiable at `x = mu`.

Anchored dynamics, for a smooth anchor `U0` and `Delta = U - U0`:

```
dX_t = -grad U0(X_t) exp(Delta(X_t)) dt + sqrt(2) exp(Delta(X_t)/2) dW_t
```

`pi ∝ exp(-U)` is invariant for any anchor (the notebook derives it from `b p = grad(D p)`),
and `U` is used only through the scalar weight `exp(Delta)` — `grad U` is never evaluated.

Anchors compared, against a subgradient-Langevin baseline: `U0 = 0`, quadratic
`U0 = (x-mu)^2/(2 tau^2)`, and pseudo-Huber `U0 = sqrt((x-mu)^2 + delta^2)/b`.

Experiments:

1. **Laplace on R** — W1 curves and terminal densities for `b ∈ {0.5, 1, 2}` (the univariate
   counterpart of the notebook's dim1/dim2/dim3 panels), plus an explicit demonstration that the
   zero anchor diverges when the state space is unbounded.
2. **Truncated Laplace on `[-R, R]`** — the projected/constrained setting of the Gaussian
   notebook, with reference samples by rejection; all four methods, including the zero anchor,
   which is well behaved on a compact set.
3. **Discretization bias vs stepsize** — W1 at stationarity over `eta ∈ [0.005, 0.2]`, 3 runs each.

Headline results: all anchors reproduce the target without `grad U`; the quadratic anchor is the
fastest to converge on the slowest-mixing target (`b = 2`: W1 0.147 vs 0.265 for the subgradient
baseline at iteration 2000); the pseudo-Huber anchor keeps `Delta` bounded, needs no clamping, and
is the most accurate method in the truncated experiment and at the largest stepsizes.

## Non-reversibility: constant skew J vs J = 0

`anchored_langevin_nonreversible.ipynb` adds a constant skew-symmetric `J` (the Gaussian notebook's
`J_fun`) to the anchored dynamics:

```
dX = -(I + J) grad U0(X) exp(Delta(X)) dt + sqrt(2) exp(Delta(X)/2) dW
```

The notebook proves invariance is unchanged for any `a` (the extra flux is `J grad(exp(-U0))`, and
`div(J grad phi) = 0` for constant skew `J`), so `J` alters the rate only.

Two facts shape the experiments. First, **in d = 1 every skew-symmetric matrix is zero**, so a
constant-J scheme applied to a univariate target *is* the J = 0 scheme — the state space has to grow.
Second, the non-reversible drift is `-J grad U0 exp(Delta)`, so it acts through the **anchor's**
gradient: with `U0 = 0` the value of `a` is irrelevant.

1. **Product Laplace in d = 3**, `b = (0.5, 1, 2)`: `{subgradient LMC, anchored} x {J = 0, J_a}`, plus a
   sweep over `a` and an `eta`-refinement check.
2. **Univariate Laplace via a 2D lift**: sample `pi(x1) pi(x2)` with skew coupling, read the x1 marginal.

Results: `J` accelerates the slow coordinate (d = 3, coordinate 3: W1 0.212 -> 0.133 at `a = 2`), leaves
the stationary law untouched, and costs discretization accuracy on the fast coordinates at large `a` —
recoverable by shrinking `eta`. The lift gives the biggest win on a genuinely univariate target:
W1 0.272 -> 0.071 at `a = 2`, for one extra scalar per particle.

Figures are written to `figures/`. Requires `numpy`, `scipy`, `torch`, `matplotlib`, `seaborn`.

## Multivariate Laplace

`anchored_langevin_multivariate.ipynb` repeats the J = 0 vs constant-J comparison on genuinely
multivariate (non-separable) Laplace targets, with `Sigma = D C D`, `D = diag(0.5, 1, 2)`, `rho = 0.6`:

- **Target A**, elliptical Laplace: `U(x) = sqrt(x' inv(Sigma) x)` — kink at the single point 0.
- **Target B**, correlated l1 Laplace: `U(x) = ||inv(L) x||_1` — kinks on d hyperplanes; reduces to the
  product Laplace when Sigma is diagonal.

Both have exact reference samplers (Gamma radius on the sphere; iid Laplace pushed through L), so no
rejection step is needed. Experiments: J = 0 vs J_a on both targets, a sweep over a, a fixed-physical-time
control showing the speed-up is not a discretization artifact, and a dimension scan d = 2, 3, 5, 10.

Results: J gives ~1.5x on the slow coordinate for both targets, leaves the stationary law untouched
(KS 0.010-0.032 at stationarity), and the gain survives eta-refinement at fixed horizon. It shrinks with
dimension (2.5x at d = 2 down to 1.1x at d = 10) because the tridiagonal J_a only couples neighbouring
coordinates. The smoothed anchors keep exp(U - U0) within [0.83, 1.0], so no clamping is required.

## Heavy-tailed Gibbs target

`anchored_langevin_heavy_tailed.ipynb` repeats the experiments with
`U(x) = iota * log(1 + ||x||^2)`, `iota > 1 + d/2`, so `pi(x) = (1+||x||^2)^{-iota}` has polynomial tails.
This `U` is smooth, so the anchor's role shifts from smoothing a kink to re-timing the tail: taking
`U0 = c log(1+||x||^2)` gives drift `-2c x (1+||x||^2)^{iota-c-1}`, and `c = iota-1` makes it linear.

The target is the multivariate t with `nu = 2 iota - d`, sampled exactly as `Z/sqrt(G)`; `nu > 2` is
precisely the stated condition `iota > 1 + d/2`.

Results: the anchor exponent dominates — `c = iota-1` reaches the sampling floor while plain Langevin is
2x away, with `q99(||x||)` 5.25 vs 3.63 against a target 5.40. J does little on the isotropic target (no
slow/fast directions) and hurts once the anchor already works; on an anisotropic version it helps again,
best combination beating plain Langevin by 4.5x. A fixed-physical-time control shows the J gain is real
for plain Langevin but not for the tail-accelerated anchor, so the two mechanisms are partly redundant.

## Constrained sampling on the ball, with a state-dependent J

`anchored_langevin_ball_constrained.ipynb` samples on `K = {x in R^3 : ||x||_2^2 <= 1}` by projection,
comparing `J = 0` against the state-dependent axial field `J_s(x) w = s (x cross w)`, and **tunes s**. The
constant skew `J_a` is excluded here because it violates the boundary condition below;
`anchored_langevin_paper_J.ipynb` measures what that costs.

State dependence changes the dynamics: the invariant form becomes

```
dX = e^Delta [ -(I + J(x)) grad U0 + div J(x) ] dt + sqrt(2) e^(Delta/2) dW
```

and the constraint adds a second requirement, `J(x) nu(x) = 0` on the boundary, so the skew drift is
tangential. The axial field satisfies both: div J = 0 to machine precision, and J(x)x = 0 identically
(max |J nu| = 5.6e-16 on the sphere).

Because J cannot move the invariant law, the tuning target is a rate. Two are reported, both averaged over
7 seeds: W1 at a fixed iteration budget, and tau(eps), the iterations needed to reach accuracy eps and stay
there.

Tuning a constant s exposes a trade-off: tau(0.03) falls monotonically (624 -> 208 on Target A as s goes
0 -> 20) while the stationary W1 is flat to s ~ 6 then climbs (0.0126 -> 0.0189). So s = 6 is free but only
1.7x, and s = 12 is 2.4x but 18% worse at stationarity.

Annealing removes the trade-off. Taking `s(k) = s0 max(0, 1 - k/K)` with s0 = 20, K = 800 is admissible for
the same reason a constant is — div J = 0 and J nu = 0 hold for every value of s, so every step preserves
the same target and the chain is merely non-homogeneous — and after iteration K the scheme is J = 0, so it
inherits its accuracy exactly.

Results: the annealed field beats J = 0 at every threshold on both targets and ties it at stationarity.
Target A tau(0.05/0.035/0.03) = 164/194/216 against 474/571/624, a 2.9x speed-up, W1 at 300 iterations
0.0186 vs 0.0984 (5.3x); Target B 117/141/159 against 469/559/598, up to 4.0x, W1 at 300 iterations 0.0157
vs 0.0966 (6.2x). Stationary W1 is 0.0125 vs 0.0126 and 0.0131 vs 0.0129, boundary mass and max KS
unchanged. The gaps are 10-30 seed-standard-deviations wide.

The constant-s penalty is discretization, not bias, which is what licenses the schedule: at fixed physical
time the ratio of s = 12 error to s = 0 error falls 1.62 -> 1.40 -> 1.12 -> 0.99 as eta is refined 8x, then
sits inside the seed noise. The projection atom is a separate artefact present at every s including zero,
scaling like sqrt(eta) (11.8% -> 6.1% -> 3.1%).

## Diagnosing "J does not accelerate", and a rebuilt integrator

`anchored_langevin_ball_optimized.ipynb` takes the configuration in which the skew field appears to *hurt*
(R = 1.5, eta = 5e-3, delta = 0.2: J = 0 gives W1 [0.0185 0.0283 0.0238], J_s at s = 4 gives
[0.0185 0.0305 0.0334]) and isolates the two independent causes, then fixes both.

Cause 1 is the metric. Those numbers are `W1[-100:].mean()`, the error at stationarity, and J_s leaves pi
invariant by construction — it changes the rate, not the fixed point, so it can only add discretization
there. A rate has to be read in the transient or through tau(eps).

Cause 2 is the integrator. `J_s(x) g = s (x cross g)` is tangential to the sphere, but Euler follows the
tangent line, inflating the radius by O((eta s)^2) per step. Measured directly, one Euler skew step from the
boundary leaves ||x||/R at 1.00039 (s = 4) and 1.02489 (s = 32); projection absorbs the excess, so boundary
mass climbs 7.4% -> 42.1% and the error follows. At eta = 1e-3 this is 25x smaller and invisible.

Three fixes:

1. integrate the skew part as an exact rotation (Rodrigues, Strang-split) — ||x|| preserved to 3.6e-15, so
   boundary mass becomes flat in s (7.4% -> 6.4% at s = 32, against Euler's 42.1%);
2. reflect at the boundary instead of projecting — removes the projection atom, 7.5% -> 0.24% against a
   target 0.22%, and drops the stationary W1 from 0.0218 to 0.0130, exactly the sampling floor;
3. anneal s(k) = 16 max(0, 1 - k/200).

At s = 0 the two integrators are bit-identical, which is the control.

Results at the same eta: Target A tau(0.06/0.04/0.025) = 66/74/352 against 286/362/1776 for the original
(4.3x/4.9x/5.0x), and it reaches 0.018 in 1530 iterations where the original cannot reach it at any
iteration count; Target B 52/74/102 against 320/400/1012 (6.2x/5.4x/9.9x). Stationary W1 sits on the floor
on both, max KS improves (0.0160 vs 0.0224, 0.0230 vs 0.0260). The annealed chain and the J = 0 chain
coalesce under common noise (max |x_ann - x_0| falls 1.72 -> 0.0043 over 2000 steps), so the schedule
provably costs nothing at stationarity.

Caveat: fix 2 does most of the accuracy work and fixes 1 and 3 do the speed work — reflection alone with
J = 0 already reaches the floor. The radial reflection map is not exactly measure-preserving (Jacobian
((2R-r)/r)^(d-1)); its error is O(eta) and the refinement study shows no plateau, but it should be
rechecked at much larger eta.

## Closed-form anchored U0, and where J belongs

`anchored_langevin_closedform_U0.ipynb` runs the specified scheme

```
x <- Pi_K( x - eta a(x) grad U0(x) + eta alpha e^U(x) J(x) grad psi(x) + sqrt(2 eta a(x)) xi ),
a(x) = e^{U(x) - U0(x)},  U = f + g,  U0 = f + g0,  g0 = E[g(x + mu xi)]
```

with g0 evaluated in **closed form** by the piecewise-quadratic lemma rather than by an inner Monte Carlo
step, for the Lasso, MCP and SCAD regularizers. One generic implementation covers all three; only the
breakpoint/coefficient table changes. Verified: the closed form matches a 2e6-sample MC estimate to 1.9e-4,
which is within MC's own standard error, and p0' matches d/dx p0 to 8e-11.

**Where J attaches is structural.** Because psi = (lam - G)h has grad psi = -2xh parallel to nu on the
boundary, `(J grad psi).nu` is proportional to `(Jx).x = 0` for *any* skew J — so a constant skew is
admissible automatically (max |(J grad psi).nu| = 4.4e-16), and `div(J grad psi) = 0` exactly for any
constant skew and any psi. This settles a question the earlier notebooks left open: the constant skew was
inadmissible there only because it acted on grad U0 (mean |(J grad U0).nu| = 0.703). Corollary: the axial
field J(x)w = grad psi x w cannot be used in this formulation, since J grad psi = grad psi x grad psi = 0
identically. The two constructions are alternatives, not a pair.

**The rotation axis is the binding constraint.** A constant skew in R^3 rotates only the plane orthogonal
to its axis. At alpha = 0.5 on Lasso, tau(0.06) is 960 (1.1x) with the axis along the slowest eigenvector of
Sigma, 830 (1.3x) for the tridiagonal J, and 417 (2.5x) with the axis along the *fastest* eigenvector —
which leaves the rotation plane spanned by the two slowest directions. Design rule: take the axis to be the
smallest-variance eigenvector of the target covariance.

Results with the aligned axis and alpha annealed 2 -> 0 over 800 iterations, against 1.2-1.3x for the
unaligned axis: Lasso tau(0.06) 327 vs 1057 (3.2x), MCP 247 vs 1057 (4.3x), SCAD 313 vs 1007 (3.2x), and
4-6x lower error at a 300-iteration budget. Stationary W1, boundary mass and max KS are unchanged or
slightly better on all three.

Recorded negative result: setting h = e^{-U0} in psi to cancel the e^U factor is admissible and does shrink
the drift's dynamic range (58x -> 23x), but is consistently worse than h = 1.

## J = 0 vs a state-dependent J, on each regularizer

`anchored_langevin_statedep_J.ipynb` compares alpha = 0 against a genuinely state-dependent J(x) on Lasso,
MCP and SCAD, using the same closed-form anchored U0 and the same scheme as above.

Because J acts on grad psi (not grad U0), the usual axial field is unavailable: k = grad psi gives
J grad psi = grad psi x grad psi = 0 identically (measured median ||J grad psi|| = 0.0000). The fix is to
keep the axial form J(x)w = k(x) x w but generate it from a *different* potential, k = grad chi. Then
`div(k x grad psi) = grad psi . (curl k) - k . (curl grad psi)` and both terms vanish because both fields
are curl-free, so invariance holds exactly for any chi; tangency on the boundary is automatic as before.
The family is graded by chi: linear chi gives the constant skew, chi = 1/2 x'Sinv x gives a linear k, and
chi = U0 gives a fully state-dependent k that is free (grad U0 is already computed). Verified by autograd:
max |curl k| <= 5.6e-17, max |div(J grad psi)| <= 8.9e-16, max |(J grad psi).nu| <= 8.9e-16 on dK.

Results with k = grad U0 and alpha annealed 2 -> 0 over 800 iterations, against J = 0:

| | tau(0.06) | tau(0.04) | W1 @ 300 | W1 stationary | max KS |
|---|---|---|---|---|---|
| Lasso, J = 0 | 1057 | 1300 | 0.2564 | 0.0203 | 0.0310 |
| Lasso, state-dep | 380 (2.8x) | 567 (2.3x) | 0.0714 (3.6x) | 0.0162 | 0.0253 |
| MCP, J = 0 | 1057 | 1310 | 0.2603 | 0.0216 | 0.0305 |
| MCP, state-dep | 340 (3.1x) | 440 (3.0x) | 0.0694 (3.8x) | 0.0187 | 0.0272 |
| SCAD, J = 0 | 1007 | 1207 | 0.2527 | 0.0188 | 0.0315 |
| SCAD, state-dep | 343 (2.9x) | 530 (2.3x) | 0.0695 (3.6x) | 0.0154 | 0.0235 |

Max KS improves on all three and boundary mass is unchanged, so the gain is purely a rate effect. The three
regularizers behave almost identically, as expected: the skew field acts through grad psi and the geometry
of K, neither of which knows anything about g.

Qualification: the aligned *constant* skew is still slightly better (3.2x/4.3x/3.2x), so on this target
axis alignment, not state dependence, is what buys the acceleration. The state-dependent field's appeal is
that it needs no spectral information about the target, only grad U0.

## Unconstrained: J = 0 vs a constant J, on each regularizer

`anchored_langevin_unconstrained_J.ipynb` runs the unconstrained scheme (no projection) and compares
alpha = 0 against a constant skew J on Lasso, MCP and SCAD, with the same closed-form anchored U0.

**Removing K removes psi, and the naive replacement diverges.** Invariance is not the obstacle: for a
constant skew, div(J grad psi) = sum_ij J_ij d_i d_j psi = 0 for any psi, and there is no boundary
condition. Usability is. On K the factor e^U is bounded; on R^3 it is not -- over the Lasso target e^U has
median 7.4, q99 1.7e3, max 4.9e5. Carrying over grad psi = -x gives a drift ~ e^U x that overflows to inf
within 400 steps. Tempering fixes it: psi = -e^{-U0} gives grad psi = e^{-U0} grad U0, so
`alpha e^U J grad psi = alpha a(x) J grad U0` (bounded, a in [0.507, 1]) and the update collapses to
`x <- x - eta a (I + alpha J) grad U0 + sqrt(2 eta a) xi` -- anchored non-reversible Langevin, with J back
on grad U0. The two formulations are therefore not in conflict: on a constrained set J must act on grad psi
to be tangential, and unconstrained the only usable psi puts it back on grad U0.

Results at a 6000-iteration horizon (the unconstrained target is much wider, ||x|| reaching ~5 against 1.5,
so 2000 iterations is not enough for J = 0 to converge):

| | tau(0.15) | tau(0.10) | tau(0.06) | W1 @ 500 | W1 @ 6000 |
|---|---|---|---|---|---|
| Lasso, J = 0 | 1027 | 1497 | 2293 | 0.2549 | 0.0238 |
| Lasso, J const | 613 (1.7x) | 770 (1.9x) | 1030 (2.2x) | 0.1447 | 0.0232-0.0246 |
| MCP, J = 0 | 1460 | 2197 | 3543 | 0.3209 | 0.0271 |
| MCP, J const | 953 (1.5x) | 1317 (1.7x) | 1857 (1.9x) | 0.2325 | 0.0296 |
| SCAD, J = 0 | 1070 | 1543 | 2277 | 0.2685 | 0.0207 |
| SCAD, J const | 713 (1.5x) | 1000 (1.5x) | 1567 (1.5x) | 0.1914 | 0.0236 |

Best gains 2.2x / 1.9x / 1.5x, against 2.8-3.1x for the same comparison on the ball -- removing the
constraint *reduces* what the skew field buys, because circulation along level sets is most valuable
exactly where a wall is slowing the reversible dynamics. Axis alignment matters as before (tau(0.06) on
Lasso: 1810 tridiagonal vs 1030 aligned).

Two qualifications: annealing alpha is no longer the right default (constant alpha = 2 wins at the tight
thresholds on all three, since there is no boundary overshoot to avoid), and there is a small stationary
cost on MCP and SCAD (W1 0.0271 -> 0.0296, 0.0207 -> 0.0236) -- ordinary discretization from the larger
drift, so the unconstrained win should be taken at a finite budget rather than read at stationarity.

## The paper's J construction

`anchored_langevin_paper_J.ipynb` implements the skew field of *Accelerating Constrained Sampling: A Large
Deviations Approach* (Wang, Tu, Wang, Zhu) inside the anchored dynamics. Their recipe: for K = {g <= lam},
take `psi = (lam - g) h`, `k = grad psi`, `J(x) w = k(x) x w`. Curl-free gives their Assumption 3
(`div J = 0`); `grad psi = -h grad g` on the boundary gives Assumption 2 (`J n = 0`).

Three instances are run and verified (max |curl k| <= 1.3e-9, max |J(x)n| <= 4.6e-16 on the boundary,
against 1.41 for the constant J_a): the ball with h = 1 (their Eq. 3.2), the ball with h = 1 + ||x||^2, and
the smoothed l_p ball with p = 4, eps = 0.2, lam = 1 (their Eq. 3.3).

Results: the inadmissible constant field degrades monotonically with strength (2.4x on the ball, 3.0x on the
l_p set) while every admissible field is flat; and at the paper's own strength s = 5 the field reaches the
sampling floor in roughly 400 iterations against 2000-3000 for J = 0, a 4-5x reduction in iteration
complexity. The effect is entirely in the transient: at iteration 3000 all runs read 0.013-0.015.

## Constrained Bayesian linear regression

`anchored_langevin_bayes_linreg.ipynb` runs the paper's Section 3.2 experiment: their Eq. (3.8) data
(n = 1e5, m = 50, x* = [1,-0.7,-0.5] with ||x*|| = 1.32 > 1, so the constraint is active), their J_a (a = 1)
and J_s (s = 5), their stepsize 1e-4, on the ball and the smoothed l_p ball.

The paper writes the target as a sum over data but uses the average gradient; the notebook makes the
implied temperature explicit (beta = 32) and shows the whole range — beta = 1 washes the data out, beta = n
is a point mass.

Paper-matching figures: prior + three posteriors with the constraint and x*, and MSE vs iteration.
New: W1 against an exact rejection-sampled posterior, the beta sweep, minibatch vs full gradient, a
stepsize-stability study, and a constrained Bayesian Lasso where the anchoring actually does work.

Results: at the paper's 300-iteration horizon both skew fields accelerate (MSE 0.545 for J_a and 0.627 for
J_s against 0.751 for PSGLD, mean of 5 runs); at stationarity MSE cannot separate them (all within 0.5% of
the floor) because x* lies outside K, while W1 and boundary mass can. J_a is 50x off on two coordinates. The stability study separates the
two error types: J_a's error is flat in eta (0.096 -> 0.092, irreducible, from J n != 0) while J_s's falls
5x (0.297 -> 0.058, pure discretization) — with a crossover where at eta = 1e-3 the admissible field is
worse than the inadmissible one, 87% of its mass pinned to the boundary by tangential Euler overshoot.

## Constrained Bayesian logistic regression

`anchored_langevin_bayes_logistic.ipynb` runs the paper's Section 3.3 synthetic experiment: their Eq.
(3.13) data (n = 2000, X ~ N(0, 2I), 20% test split), their parameters (eta = 1e-4, m = 50, 1000
iterations, a = 1, s = 10), on the ball and the smoothed l_p ball. beta_* is not stated in the paper; the
notebook uses their linear-regression x_* and reports the resulting Bayes ceiling.

The paper uses accuracy because W1 "is not practical" for logistic regression. In d = 3 it is: a
1.9M-point grid over K gives a reference stable to 4 decimals, so both metrics are reported.

Results: this does NOT reproduce the paper's ranking. PSGLD sits on the reference (test accuracy 0.754 vs
0.754, W1 0.003-0.012) while both skew fields are worse, on both constraint sets; rescaling beta_* to raise
the ceiling to 0.91 does not flip it. Accuracy is nearly blind as a diagnostic - J_a is 30x worse in W1 yet
scores within one standard deviation on accuracy - and it saturates in ~150 iterations. The stepsize study
repeats the linear-regression split: on the exact posterior the paper's s = 10 collapses to chance (0.505,
100% of mass on the boundary), refining eta recovers J_s but not J_a.
