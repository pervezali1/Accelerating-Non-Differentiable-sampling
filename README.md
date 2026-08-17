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
comparing `J = 0`, the constant skew `J_a`, and the state-dependent axial field `J_s(x) w = s (x cross w)`.

State dependence changes the dynamics: the invariant form becomes

```
dX = e^Delta [ -(I + J(x)) grad U0 + div J(x) ] dt + sqrt(2) e^(Delta/2) dW
```

and the constraint adds a second requirement, `J(x) nu(x) = 0` on the boundary, so the skew drift is
tangential. The axial field satisfies both (div J = 0 to machine precision, J(x)x = 0 identically); the
constant field satisfies the first and violates the second.

Results: at strength 4 the constant field inflates coordinate-2 W1 by 4x (0.075 vs 0.019) and adds boundary
mass (8.1% vs 6.1%), while the axial field is indistinguishable from J = 0. Sweeping the strength, the
constant field degrades monotonically (W1 0.0136 -> 0.0405) and the axial field does not (0.0136 -> 0.0119).
The projection atom is a separate, benign error scaling like sqrt(eta); the boundary-condition violation
plateaus under stepsize refinement and does not go away.

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
