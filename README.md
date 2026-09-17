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

`results/` holds the tables and figures produced by `python main.py`
(≈ 8 minutes: 20 000 iterations, 5 000 burn-in, thin 5, 4 chains, 5 values of
`alpha`, plus the sensitivity sweep, the tight variant and the validation
target).

### Non-reversibility accelerates mixing, monotonically in `alpha`

Identical seeds, step size and starting points for every row; only `alpha`
changes.

| `alpha` | min ESS | median ESS | min ESS/s | median ESS/s | max split R-hat | median IAT (iters) | MSJD |
|---|---|---|---|---|---|---|---|
| 0.0  | 365.2 | 461.8 | 24.4 | 30.8 | 1.0084 | 130 | 1.702e-3 |
| 0.1  | 368.7 | 464.4 | 23.8 | 30.0 | 1.0079 | 129 | 1.703e-3 |
| 0.25 | 382.1 | 471.5 | 24.9 | 30.7 | 1.0068 | 127 | 1.707e-3 |
| 0.5  | 405.3 | 615.7 | 28.6 | 43.5 | 1.0065 |  97 | 1.720e-3 |
| 1.0  | **442.3** | **887.1** | **30.2** | **60.6** | 1.0071 | **68** | 1.775e-3 |

Going from the reversible baseline to `alpha = 1` gives **+21 % minimum ESS**
and **+92 % median ESS** at essentially unchanged cost per iteration, halving
the integrated autocorrelation time. Mean squared jumping distance moves by
only 4 %, so the gain does *not* come from taking larger steps — it comes from
the direction of the added drift. The autocorrelation plot makes the mechanism
visible: at `alpha = 1` several coordinates show autocorrelation dipping
*below zero* around lag 5–15, the oscillatory decay characteristic of a
rotational drift.

Posterior summaries are unaffected, as they must be — the added term does not
change the invariant measure. All five values of `alpha` give the same test
ROC-AUC (0.7967), the same confusion matrix to within one or two cases, and
`||posterior_mean - beta_true||_2 = 0.329` throughout (that residual is mostly
LASSO shrinkage from `lambda_lasso = 10`, not sampling error).

### Step-size sensitivity (`h`, `h/2`, `h/4`)

Nothing diverges at any step size and the `alpha` ordering is preserved in
median ESS/s at every one. Note that this sweep holds the *iteration count*
fixed, so smaller steps simply explore less: at `h/4` the chains have not
converged within 8 000 iterations (max split R-hat ≈ 1.09), which makes the
minimum-ESS column there unreliable. It is a stability check, not an accuracy
comparison — for that, see the time-matched validation below.

### The recommended constraint is inactive

With `radius_budget = 4.5` the threshold is `Lambda_constraint ≈ 9.6466` while
`g(beta_true) ≈ 4.178`, and the posterior concentrates within `O(n^{-1/2})` of
`beta_true`. Observed `g(w)` stays near 4.4, so the **projection frequency is
exactly 0** and the distance-to-boundary diagnostics are flat. Those settings
are used verbatim for the primary experiment, as specified.

To make the projection machinery observable, `main.py` also runs a
**tight-constraint variant** (`radius_budget = 1.45`, everything else
unchanged), where ~56 % of iterations are projected and the chain genuinely
rides the boundary. The acceleration survives the active constraint, and is if
anything cleaner:

| `alpha` | min ESS | median ESS | max split R-hat | projection freq. |
|---|---|---|---|---|
| 0.0 | 405.7 | 542.5 | 1.0117 | 0.559 |
| 0.5 | 454.3 | 629.3 | 1.0061 | 0.560 |
| 1.0 | **461.6** | **866.0** | **1.0050** | 0.561 |

`beta_true` needed no rescaling: `g(beta_raw) ≈ 4.178` is already below the
midpoint level `≈ 4.874`, so the bisection returns `t = 1`. The bisection is
implemented and is exercised by a test on a vector that does violate the bound.

### Validation against an exact reference

On the weighted-L1 target `U(x) = sum_i omega_i |x_i|` restricted to the same
`K`, exact independent draws come from rejection sampling
(`x_i ~ Laplace(0, 1/omega_i)`, keep if `g(x) <= Lambda_constraint`).
Acceptance is **0.764**, so the reference sample is large and trustworthy, and
the truncation genuinely bites (coordinate 0 has variance 1.41 against 3.13
unconstrained).

Step-size runs are **time-matched** — halving `h` doubles the iteration count,
burn-in and thinning — so bias is not confounded with reduced exploration. The
comparison reports the Monte-Carlo standard error beside every discrepancy.

| run | max abs. mean error | in MC SE | max W1 | energy distance | projection freq. |
|---|---|---|---|---|---|
| `alpha=0`, `h`   | 0.0698 | 1.9 | 0.0698 | 0.0063 | 0.037 |
| `alpha=0`, `h/2` | 0.0577 | 2.2 | 0.0640 | 0.0044 | 0.027 |
| `alpha=0`, `h/4` | 0.0371 | 1.6 | 0.0524 | **0.0026** | 0.018 |
| `alpha=0.5`, `h`   | 0.0616 | 1.7 | 0.0616 | 0.0066 | 0.038 |
| `alpha=0.5`, `h/2` | 0.0592 | 2.1 | 0.0616 | 0.0045 | 0.027 |
| `alpha=0.5`, `h/4` | 0.0323 | 1.7 | 0.0523 | **0.0028** | 0.018 |

Coordinate means agree to within about one to two Monte-Carlo standard errors
at every step size, and the empirical CDFs lie on top of the reference. The
multivariate energy distance falls monotonically, 0.0063 → 0.0044 → 0.0026 as
`h` is quartered: that residual is the Euler–Maruyama and projection
discretisation bias, and it vanishes with `h` as it should. `alpha` does not
affect accuracy, only efficiency — which is exactly the claim being tested.

### Reproducibility

Two independent full runs of `main.py` produced **bit-identical** ESS, R-hat
and MSJD for every `alpha`; only wall-clock timings differed.

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
