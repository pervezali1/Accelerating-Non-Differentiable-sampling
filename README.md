# Accelerating Non-Differentiable Sampling

A complete, reproducible implementation of a **constrained Non-Reversible
Anchored Langevin** algorithm, applied to a synthetic Bayesian
logistic-regression problem with an L1 (LASSO) prior and a smoothed
l^p constraint.

The point of the method is that the sampled target is
**non-differentiable** — it contains `lambda_lasso * sum_j |w_j|` — yet the
sampler only ever evaluates the gradient of a **smooth anchor**. A
state-dependent anchor coefficient `a(w)` corrects the discrepancy exactly, and
a divergence-free skew-symmetric field `J(w)` adds a non-reversible drift that
accelerates mixing without changing the invariant measure.

---

## 1. The algorithm

### Target (non-differentiable)

```
U(w) = sum_i [ softplus(X_i @ w) - y_i * (X_i @ w) ]      # sum, not mean
       + w[0]^2 / (2 * sigma_intercept^2)                 # weak Gaussian prior on the intercept
       + lambda_lasso * sum_{j=1}^{8} |w[j]|              # intercept NOT penalised
```

### Smooth anchor

```
U0(w) = ... same likelihood and intercept prior ...
        + lambda_lasso * sum_{j=1}^{8} sqrt(w[j]^2 + delta_anchor^2)

grad_U0    = X.T @ (sigmoid(X @ w) - y) + grad_prior
grad_prior[0] = w[0] / sigma_intercept^2
grad_prior[j] = lambda_lasso * w[j] / sqrt(w[j]^2 + delta_anchor^2),  j >= 1
```

### Anchor coefficient

```
log_a(w) = lambda_lasso * sum_{j=1}^{8} ( |w[j]| - sqrt(w[j]^2 + delta_anchor^2) )
a(w)     = exp(log_a(w))
```

Computed **directly from the penalty difference**, never as `exp(U)/exp(U0)`
(which would overflow for a 2000-term sum-likelihood). Because
`0 <= sqrt(t^2 + delta^2) - |t| <= delta`,

```
-8 * lambda_lasso * delta_anchor <= log_a(w) <= 0
exp(-8 * lambda_lasso * delta_anchor) <= a(w) <= 1
```

and these bounds are asserted in the test suite.

**Why this is exact.** Note `log_a = U - U0`, i.e. `a = exp(U - U0)`. The
diffusion `dw = -a(w) grad_U0(w) dt + sqrt(2 a(w)) dW` has stationary density
`pi ∝ exp(-U0)/a = exp(-U)`: with `a pi ∝ exp(-U0)` the Fokker–Planck flux is
`-b pi + grad(a pi) = a grad_U0 pi - grad_U0 a pi = 0`. So only the *smooth*
gradient is ever evaluated, while the *non-differentiable* target is sampled.

### Constraint

```
g(w)              = sum_{i=1}^{d} (w[i]^2 + epsilon_constraint^2)^(p_constraint/2)
K                 = { w : g(w) <= Lambda_constraint }
Lambda_constraint = d * epsilon_constraint**p_constraint + radius_budget**p_constraint
psi(w)            = Lambda_constraint - g(w),      grad_psi = -grad_g
```

### Non-reversible field `J(w)`

The `d = 9` coordinates are split into disjoint triples `(0,1,2)`, `(3,4,5)`,
`(6,7,8)`. For `k = grad_psi[(a,b,c)]` the cross-product matrix

```
[[  0, -k3,  k2],
 [ k3,   0, -k1],
 [-k2,  k1,   0]]
```

is inserted into rows/columns `(a,b,c)`. Three properties hold **exactly**
(verified numerically at run time and in the tests):

| property | why | measured |
|---|---|---|
| `J^T = -J` | each block is a hat matrix | `0.0` |
| `div J = 0` | row `a` gives `-d²psi/dw_b dw_c + d²psi/dw_c dw_b` | `0.0` |
| `J n = 0` | `[k]_x k = k × k = 0` | `~2e-16` on the boundary |

Because `div J = 0`, **no `div(J)` correction is added to the drift.**

`J` is **never normalised by a state-dependent quantity** — that would break
`div J = 0` and destroy the invariant measure. The only rescaling available is
a *constant* swirl multiplier `s` (`J -> s J`); since only the product
`alpha * s` matters, the primary experiment fixes `s = 1`.

The operator norm has a closed form (no SVD needed): `[k]_x^T [k]_x =
||k||^2 I - k k^T` has singular values `(||k||, ||k||, 0)`, and `J` is block
diagonal, so `||J(w)||_2 = s * max_block ||k_block||`. This is checked against
`numpy.linalg.norm(J, 2)` in the tests.

### One iteration

```
grad_k   = grad_U0(w_k)
a_k      = exp(log_a(w_k))
J_k      = construct_J(w_k, p_constraint, epsilon_constraint)
drift_k  = -a_k * (I_d + alpha * J_k) @ grad_k
xi_k     ~ N(0, I_d)
proposal = w_k + step_size * drift_k + sqrt(2 * step_size * a_k) * xi_k
w_{k+1}  = projection_K(proposal)
```

`alpha = 0` removes **only** the non-reversible term, leaving the reversible
anchored baseline `-a grad_U0` bit-for-bit identical (tested against an
independent reference implementation).

### Exact Euclidean projection

`projection_K(y) = argmin_z 0.5||z-y||^2  s.t.  g(z) <= Lambda_constraint`,
solved through its KKT system rather than by radial scaling:

1. If `g(y) <= Lambda_constraint`, return `y`.
2. Otherwise there is `eta > 0` with
   `z_i + eta * p * z_i * (z_i^2 + eps^2)^(p/2-1) = y_i` and `g(z) = Lambda`.
3. **Inner solve.** For fixed `eta` the scalar map is odd and strictly
   increasing (`h'(z) = 1 + eta p (z²+eps²)^(p/2-2)((p-1)z² + eps²) > 0` for
   `p >= 1`), so `|z_i|` is found by `scipy.optimize.brentq` on `[0, |y_i|]`.
4. **Outer solve.** `g(z(eta))` is decreasing; start `eta_low = 0`, grow
   `eta_high` geometrically until `g(z(eta_high)) <= Lambda`, then `brentq`.
5. Feasibility and the KKT residual are verified on every call.

A `p_constraint = 2` shortcut is included (`K` is then the Euclidean ball of
radius `sqrt(Lambda - d eps^2)` and the projection is radial). **Radial scaling
is not used for `p != 2`**; a test demonstrates it is strictly sub-optimal
there.

---

## 2. Layout

```
config.py                 all parameters, dataclasses, every fixed seed
synthetic_data.py         AR(1) predictors, standardisation, Bernoulli responses
target.py                 U, U0, grad_U0, log_a, a  (logistic + weighted-L1)
constraint.py             g, grad_g, psi, grad_psi, normal, beta_true rescaling
nonreversible_matrix.py   construct_J, operator norm, divergence, tangency
projection.py             exact KKT-based Euclidean projection onto K
sampler.py                projected non-reversible anchored Langevin
diagnostics.py            ESS / R-hat / IAT / MSJD, posterior prediction, figures
reference_validation.py   §13 weighted-L1 target vs exact rejection sampling
experiment.py             alpha comparison, step-size sensitivity, audits
main.py                   runs the entire study end to end
tests/test_correctness.py the twelve §12 checks (plus supporting tests)
```

## 3. Running it

```bash
pip install -r requirements.txt

python main.py                  # full study, writes ./results
python main.py --quick          # ~2 minute smoke test
python main.py --help           # all options

pytest tests -q                 # or: python tests/test_correctness.py
```

Everything is seeded from `MASTER_SEED = 20240917` (see
`config.describe_seeds()`), and reproducibility is asserted in the tests.

### Experimental protocol

`alpha in {0, 0.1, 0.25, 0.5, 1.0}`, four independent chains each. The target,
constraint, step size, starting points, iteration count, burn-in, thinning
interval and **all random seeds are identical across every `alpha`**, so the
comparison is exactly paired. Chain `c` always uses seed `CHAIN_SEED_BASE + c`.

The step size is derived from the curvature of `U0`,
`step_size = step_scale / L` with `L = 0.25 * lambda_max(X^T X) +
max(1/sigma^2, lambda_lasso/delta_anchor)`, and the sensitivity study repeats
the comparison at `h`, `h/2` and `h/4`.

Starting points are overdispersed around the smooth MAP (minimiser of `U0`) and
every one of them is projected onto `K`.

---

## 4. What the experiment shows

See `results/` for the tables and figures produced by `main.py`.

**Non-reversibility helps, monotonically in `alpha`.** Per-coordinate ESS and
split R-hat both improve as `alpha` grows from 0 to 1, at essentially identical
cost per iteration, so ESS-per-second improves too. Mean squared jumping
distance is nearly unchanged — the gain comes from the *directional* structure
of the added drift, not from larger steps.

**A caveat about the recommended constraint.** With `radius_budget = 4.5` the
threshold is `Lambda_constraint ≈ 9.6466` while `g(beta_true) ≈ 4.178`, and the
posterior concentrates within `O(n^{-1/2})` of `beta_true`. The constraint is
therefore **never active**: projection frequency is exactly 0 and the
distance-to-boundary diagnostics are flat. Those settings are used verbatim for
the primary experiment, as specified. To make the projection machinery
observable, `main.py` additionally runs a **tight-constraint variant**
(`radius_budget = 1.45`, everything else unchanged) in which roughly half of
the iterations are projected and the chain genuinely rides the boundary.

**`beta_true` needed no rescaling.** `g(beta_raw) ≈ 4.178` is already below the
midpoint level `≈ 4.874`, so the bisection returns `t = 1`. The bisection is
implemented and is exercised by a test on a vector that does violate the bound.

**The sampler is validated against an exact reference.** On the weighted-L1
target `U(x) = sum_i omega_i |x_i|` restricted to the same `K`, exact
independent draws come from rejection sampling
(`x_i ~ Laplace(0, 1/omega_i)`, keep if `g(x) <= Lambda_constraint`;
acceptance ≈ 0.76, so the reference sample is large and trustworthy). Step-size
runs are **time-matched** (halving `h` doubles the iteration count), and the
comparison reports the Monte-Carlo standard error next to each discrepancy —
without it, sampling noise is easily mistaken for discretisation bias. The
coordinate means agree to within about one MC standard error.

---

## 5. Correctness tests (§12)

| # | check | where |
|---|---|---|
| 1 | `g(0) = d * epsilon_constraint**p_constraint` | `test_g_at_origin` |
| 2 | `Lambda_constraint > g(0)` | `test_threshold_exceeds_origin` |
| 3 | every sampled state lies in `K` | `test_every_state_is_feasible` |
| 4 | `J(w)` is skew-symmetric | `test_J_is_skew_symmetric` |
| 5 | finite-difference `div J ≈ 0` | `test_divergence_of_J_is_zero` |
| 6 | `J(w) n(w) ≈ 0` on the boundary | `test_tangency_on_the_boundary` |
| 7 | anchor coefficient bounds | `test_anchor_bounds` |
| 8 | intercept excluded from the L1 penalty | `test_intercept_is_not_l1_penalised` |
| 9 | projection satisfies its KKT conditions | `test_projection_kkt` |
| 10 | reproducible for fixed seeds | `test_reproducibility` |
| 11 | `alpha = 0` removes only the non-reversible term | `test_alpha_zero_is_the_reversible_baseline` |
| 12 | no state-dependent normalisation of `J` | `test_no_state_dependent_normalisation_of_J` |

Supporting tests additionally cross-check the projection against SLSQP, confirm
the `p = 2` radial shortcut, show that radial scaling is *not* the projection
when `p != 2`, verify the closed-form operator norm against an SVD, check
`grad_U0` against finite differences, and confirm that only the product
`alpha * s` affects the dynamics.

---

## 6. Naming

The specification's names are used verbatim and never aliased. In particular
`Lambda_constraint` (the constraint threshold) and `lambda_lasso` (the L1 prior
strength) are distinct throughout.

| symbol | meaning |
|---|---|
| `p_constraint` | exponent of the smoothed l^p constraint |
| `epsilon_constraint` | smoothing parameter of the constraint |
| `Lambda_constraint` | constraint threshold |
| `lambda_lasso` | L1-prior strength |
| `delta_anchor` | smoothing parameter of the L1 anchor |
| `alpha` | non-reversibility strength |
| `step_size` | Euler step size |
