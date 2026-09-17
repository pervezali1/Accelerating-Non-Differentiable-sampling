# Non-reversible anchored Langevin with a block state-dependent skew-symmetric matrix

Synthetic Bayesian logistic regression on two constraint sets, comparing four
samplers that share one implementation and differ only in $(\rho, \alpha)$.

```
anchored_sgld.py                            reusable library (data, target, geometries,
                                            projections, anchor, J, sampler, checks, plots)
nonreversible_anchored_langevin.ipynb       the runnable notebook (clean, no outputs)
nonreversible_anchored_langevin.executed.ipynb   the same notebook with the full run's outputs
results/                                    figures (300-dpi PNG + vector PDF) and saved arrays
```

```bash
pip install numpy scipy pandas scikit-learn matplotlib jupyter
jupyter lab nonreversible_anchored_langevin.ipynb     # full run, ~6 min
NRAL_QUICK=1 jupyter lab ...                          # quick mode: 5 repeats, 300 iterations
```

Quick-mode artefacts are written with a `_quick` suffix and every printed table is
tagged `QUICK MODE`, so they cannot be confused with the full experiment. The
committed `results/` are from the **full** experiment (R = 100, 1000 iterations).

## Setup

| | |
|---|---|
| Dimension | $d = 9$ (optional $d = 3$ with one block, $s = 10$) |
| Observations | $n_{\text{total}} = 2000$, stratified 80/20 → 1600 train / 400 test |
| Mini-batch | $m = 50$, uniform **without replacement** within each iteration |
| Iterations / step | 1000 at $h = 10^{-4}$ |
| Replicates | $R = 100$ (sensitivity sweep: 20, configurable) |
| Block strengths | $s = (10, 10, 10)$, configurable independently |
| Seeds | `data_seed = 2026`, `split_seed = 2027`, `sampler_seed = 3000` |

Data: $X_j \sim N(0, 2I_d)$ (coordinate sd $\sqrt2$), $y_j = \mathbf 1\{u_j \le
\sigma(X_j^\top\beta_{\text{true}})\}$, no intercept, no standardisation. The
constraints act on the **coefficients**, not on the feature vectors. The data set
and split are frozen across every method, replicate and geometry, so the replicate
spread is sampler randomness only.

Target: uniform prior on $K$ times the logistic likelihood, with $U$ a **sum** over
the 1600 training rows and
$\widehat G_k = \frac{n_{\text{train}}}{m} X_{B_k}^\top[\sigma(X_{B_k}\beta_k) - y_{B_k}]$.

Anchor: $U_0 = U + \rho H_K$, $a = e^{-\rho H_K}$, computed exactly from the
geometry (never by exponentiating a noisy likelihood difference). $H_K \in [0,1]$
on $K$, so $\rho = \log 2$ gives $\tfrac12 \le a \le 1$.

Update:
$$\beta_{k+1} = \Pi_K\!\left[\beta_k - h a_k\bigl(v_k + \alpha J(\beta_k)v_k\bigr) + \sqrt{2ha_k}\,\xi_k\right],\quad v_k = \widehat G_k + \rho\nabla H_K(\beta_k).$$

No $\nabla a$ correction; $J$ is never rescaled beyond the constant block
strengths. Within a replicate and geometry all four methods get identical starting
coefficients, mini-batch streams and Gaussian increments, from separate spawned
sub-streams so the noise is independent of the batch.

| Method | $\rho$ | $\alpha$ |
|---|---:|---:|
| Projected SGLD | 0 | 0 |
| Non-reversible SGLD | 0 | 1 |
| Reversible anchored Langevin | $\log 2$ | 0 |
| Non-reversible anchored Langevin | $\log 2$ | 1 |

## Verified before any production run

Every item is an assertion in §7 of the notebook.

| check | result |
|---|---|
| $\nabla U$ vs finite differences | rel. error 3.4e-10 |
| mini-batch scaling, **all 56 batches enumerated** | error 4.4e-16; without the $n/m$ factor, off by 1.08 |
| $J^\top = -J$ | 0.0 exactly (both geometries) |
| $\operatorname{div} J = 0$ (finite differences) | 0.0 exactly |
| $Jn = 0$ on $\partial K$ | ≤ 8.9e-16 |
| $J\nabla H_K = 0$ | ≤ 5.7e-15 |
| matrix-free $Jv$ vs explicit matrix | ≤ 7.1e-15 |
| $\tfrac12 \le a \le 1$ on $K$ | holds |
| projection KKT residual / feasibility | ≤ 2.7e-15 / ≤ 1.1e-15 |
| vectorised bisection vs scalar `brentq` | 1.4e-15 |
| $J\nabla U_0 \ne 0$ | median norm 2248 (ball), 4996 (quartic) |
| radial scaling on the quartic set | strictly worse in **40/40** projected cases |
| **ball** $J$ on the quartic boundary | $\max|Jn| = 2.88 \ne 0$ — this is why the quartic uses $-s\nabla_{I}g$ |

## What was observed

**At $h=10^{-4}$ with $s=10$, the non-reversible methods are worse on accuracy**,
on both geometries. The two reversible methods are indistinguishable from each
other, so the anchor alone is neutral here.

| geometry | Projected SGLD | Reversible anchored | Non-rev. anchored | proj. rate (NR) | $\|\alpha Jv\|/\|v\|$ |
|---|---|---|---|---|---|
| unit ball | 0.6548 ± 0.0102 | 0.6540 ± 0.0095 | 0.6281 ± 0.0238 | 0.041 | 4.2 |
| quartic set | 0.6543 ± 0.0099 | 0.6541 ± 0.0096 | 0.5429 ± 0.0529 | 0.767 | 13.3 |

This is reported as measured. It is **not** evidence against non-reversible
sampling — it is a step-size failure specific to this modified sampler, supported
by four independent pieces of evidence:

1. **The added term dominates the update.** $\|\alpha Jv\|/\|v\|$ is 4–13, and the
   mean per-step displacement reaches 0.57 on a domain of radius ~1. The
   projection then fires on up to 77% of iterations: the chain is pinned to the
   boundary rather than circulating in the interior.
2. **Mini-batch noise is amplified.** $J$ multiplies the gradient *error* as well
   as the gradient — measured amplification ×4.5. The resulting per-step
   displacement (0.215) is ~6× the injected $\sqrt{2ha}$ increment (0.037), so the
   added term sets the noise level instead of perturbing the dynamics. The
   continuous-time invariance argument assumes the *exact* gradient and does not
   cover this.
3. **The full-gradient control isolates it.** With exact gradients the ball
   deficit vanishes entirely; what remains on the quartic is the deterministic
   oversized $\alpha Jv$ step.
4. **Refinement removes the gap.** At constant simulated time $t = kh$:

   | geometry | gap at $h$ | at $h/2$ | at $h/4$ |
   |---|---|---|---|
   | unit ball | −0.0335 | −0.0205 | −0.0071 |
   | quartic set | −0.1018 | −0.0505 | **−0.0002** |

   The block-strength sweep agrees: accuracy is at parity for $s \le 5$ (drift
   ratio ≤ 1.7, projection ≈ 0) and collapses only at $s = 10$.

**Answer to the stated question.** Which method reaches 0.64 test accuracy sooner,
in iterations and wall time? At $h = 10^{-4}$, Projected SGLD, on both geometries
(ball 20 vs 60 iterations; quartic 30 vs 245). **That conclusion does not survive
step refinement** — by $h/4$ the two are within 0.0002 accuracy of each other.

## Limitations

* **Accuracy is not posterior convergence.** These are single-iterate
  classification accuracies; they say nothing about whether $\pi_K$ was reached or
  about asymptotic variance. A method can match on accuracy and still be biased.
* **The invariance identity does not cover the algorithm.** It holds for the
  continuous-time process with exact gradients. The implementation adds finite-$h$
  discretisation bias, mini-batch gradient noise, and the projection, which places
  mass on $\partial K$ and is not a discretisation of a measure-preserving
  reflected process.
* **Bounded iterates prove nothing.** $\Pi_K$ keeps every state in $K$ by
  construction, so the absence of blow-up is not evidence of numerical accuracy.
  Projection rates and non-finite counts are reported for exactly this reason.
* **One data set, one split, one $\beta_{\text{true}}$.** Replicate spread is
  conditional on the frozen data; it excludes data-sampling variability.
* **Bands are repeat-run variability** — not confidence intervals and not
  posterior credible intervals.
* The problem is intrinsically noisy: the Bayes ceiling is ≈ 0.671 test accuracy
  and $\beta_{\text{true}}$ itself scores 0.655, so ≈ 0.654 is essentially optimal,
  not an underfit.

Nothing was tuned against test labels, the $n_{\text{train}}/m$ factor was never
removed, the drift was never clipped, and $J$ was never rescaled.

## Saved results

`results/results_d9.npz` holds every checkpoint array (train/test accuracy per
replicate, coefficient checkpoints, training loss, constraint values, anchor
values) for the main runs, the full-gradient control, the $d=3$ runs and all three
sensitivity step sizes. `results_d9_metadata.json` records the configuration,
seeds, package versions, timings, per-run diagnostics and all check outputs.
`accuracy_curves_mean_sd.csv` holds the per-checkpoint means and sample standard
deviations directly.

§11c of the notebook reloads the arrays from disk, rebuilds the result objects and
redraws a main figure, asserting the reloaded values match to 0.0 — figures are
regenerable without rerunning any sampler.


---

## Block-strength search: can non-reversible beat reversible?

`block_strength_search.py` → `results_search/`. Only the **two anchored methods**
appear (ρ = log 2 in both; α = 0 vs 1), so every comparison isolates
non-reversibility at a fixed anchor.

Both methods share starting points, mini-batch streams and Gaussian increments
within each replicate, so the test is **paired** on the per-replicate difference.
That matters: at R = 100 an unpaired comparison cannot resolve a 0.003 accuracy
difference, and a paired one can.

### On the specified design, no s wins on accuracy

Twenty configurations (s ∈ [0.25, 10] × both geometries). The best paired
t-statistic is **+1.98** — which is what you expect from the *best of twenty*
tests under the null. Every error bar crosses zero, and the sign flips between
metrics (at s = 1 on the ball the final-accuracy difference is +0.0010 while the
area-under-the-curve difference is −0.0001, t = −2.09).

The reason is structural, not a tuning failure: **accuracy has no headroom.** The
Bayes ceiling is 0.671, `beta_true` itself scores 0.655, and both methods reach
0.654. There is nothing left for a better sampler to win.

### On the specified design, s does win on the metric non-reversibility targets

Non-reversible perturbations are designed to reduce the **asymptotic variance of
ergodic averages**, not to move the invariant measure. Measuring exactly that —
the across-replicate variance of the time-averaged coefficient — gives a clean,
monotone effect:

| s | 0.25 | 1 | 2 | **3** | **4** | 5 | 7.5 | 10 |
|---|---|---|---|---|---|---|---|---|
| Var ratio NR/REV (ball) | 0.995 | 0.953 | 0.884 | 0.846 | **0.844** | 0.878 | 1.21 | 2.22 |

A **15.6 % variance reduction at s = 4**, then a sharp reversal as J's
amplification of mini-batch noise takes over (and, on the quartic set at s ≥ 7.5,
as the projection starts firing). That U-shape is the whole mechanism in one row.

### Where non-reversibility wins on accuracy

Non-reversibility buys its advantage from **anisotropy**, and the specified design
has almost none (posterior condition number **1.63**). Switching the predictors to
AR(1) with ρ_x = 0.99 — same pipeline, same marginal variance 2, only the
conditioning changes — gives condition number **1225**, and then the win appears:

| | REV | NR (s = 5) | paired diff | t |
|---|---|---|---|---|
| search seed | 0.6973 | 0.6991 | +0.0017 | +1.89 |
| held-out 101 | 0.6941 | 0.6975 | +0.0034 | +3.57 |
| held-out 202 | 0.6962 | 0.6979 | +0.0018 | +1.50 |
| held-out 303 | 0.6963 | 0.6984 | +0.0022 | +2.24 |

`s = 5` was selected on the search seed and then **confirmed on three sampler
seeds that took no part in the search**: all three positive, pooled difference
**+0.0024 (SE 0.00060, t = +4.09)**. The effect is strongest mid-trajectory
(t = +2.7 at iteration 400) and is monotone in s up to the point where the
projection rate starts to bite — consistent with the mechanism being *faster
convergence in the slow direction*, which only pays while the run is still
convergence-limited. At s = 7 the mid-run advantage is larger (t = +4.1 at
iteration 400) but has evaporated by iteration 1000, because a 32 % projection
rate costs more than the acceleration is worth.

### Exact-gradient LASSO variant, and a correction

`anchored_lasso.py` + `accuracy_curves_lasso_exact.ipynb` replace the target with
the genuinely non-differentiable LASSO potential and drop the mini-batch:

```
U(w) = sum_j [softplus(x_j.w) - y_j x_j.w] + w0^2/(2 sigma^2)   <- f, differentiable
       + lambda_lasso * sum_{j>=1} |w_j|                        <- g, NOT differentiable
U0   = f + g_delta,   a = exp(U - U0) = exp(lambda * sum(|w_j| - sqrt(w_j^2+delta^2)))
x_{k+1} = Pi_K[ x_k - eta a grad_U0 + eta alpha a J_s grad_U0 + sqrt(2 eta a) xi ]
```

Here the anchor does real work: only `grad U0` is evaluated, exactly (no
subsampling), while the invariant measure carries the true kinked `g`. Column 0
of `X` is now an intercept, because `U` prices `w0` separately. Run over the
Euclidean ball and the smoothed **L1** ball (Lambda = 3.7). `SmoothLpBallGeometry`
handles any p >= 1 and reproduces the independently verified quartic set at p = 4
and the L1-smooth ball at p = 1, both to ~1e-16.

**On accuracy: the two methods are statistically indistinguishable on both
constraint sets**, at every swept block strength, on both the specified and an
ill-conditioned design. Held-out confirmation fails on both (pooled t = -0.71
and -0.89). That is a property of the observable, not of the sampler: accuracy
is saturated here (Bayes ceiling ~0.67, both methods reach it), so it cannot
resolve a sampler improvement.

**On the observable non-reversibility actually targets, it wins decisively.**
Non-reversible perturbations are designed to reduce the asymptotic variance of
ergodic averages, not to move the invariant measure. Measuring that
(`ergodic_variance_search.py`), with exact gradients:

| s | 0.5 | 1 | 2 | 3 | 5 | **7** | 10 |
|---|---|---|---|---|---|---|---|
| Var ratio NR/REV, unit ball | 0.978 | 0.927 | 0.798 | 0.694 | 0.584 | 0.539 | **0.537** |
| Var ratio NR/REV, **L1 ball** | 0.853 | 0.668 | 0.517 | **0.483** | 0.655 | 4.36 | 4.32 |

Both geometries win, and the **L1 ball wins fastest** — it reaches its optimum at
s = 3, where the unit ball still needs s = 10:

| | selected s | held-out ratios | mean | reduction | all below 1 |
|---|---|---|---|---|---|
| **L1-smooth ball** | 3 | 0.520, 0.516, 0.553, 0.526 | **0.529** | **47.1%** | yes |
| unit ball | 10 | 0.519, 0.501, 0.567, 0.560 | 0.537 | 46.3% | yes |

Past its optimum the L1 ball degrades sharply (s >= 7 drives the projection rate
to 0.31 and the variance ratio above 4), which is the same oversized-step
mechanism seen elsewhere.

**Correction to the earlier unit-ball result.** The confirmed win reported in
`block_strength_search.py` (pooled held-out +0.00244, t = +4.09) was measured
with mini-batch gradients. Holding target, design, s, seeds and geometry fixed
and changing only the gradient:

| gradient | pooled held-out | t | all positive |
|---|---|---|---|
| mini-batch (as reported) | +0.00244 | +4.09 | yes |
| exact | -0.00000 | -0.00 | no |

That win was the stochastic gradient interacting with J, not non-reversibility.
The earlier notebooks stand with their mini-batch settings, but the claim should
be read with this correction attached.

**What exact gradients did fix.** With mini-batch gradients, s = 10 on the
Euclidean ball drove the projection rate to 0.04 and cost 0.026 accuracy; with
exact gradients the same s leaves the projection rate at 0.000 and accuracy
within noise — direct confirmation that the damage was J amplifying the gradient
*error*. What survives at large s on the l^p ball is the separate deterministic
mechanism, an oversized `alpha J grad_U0` step.

### Varying lambda_lasso: the anchor-free control and beyond

`lambda_zero_experiment.py` -> `results_lambda0/`. At `lambda_lasso = 0` the
non-differentiable part vanishes, `U = f` is smooth, `a(w) = 1` identically, and
the update collapses to plain projected Langevin. That is asserted, not assumed:
the reversible chain is **bitwise identical** to `x - eta*grad_f + sqrt(2 eta) xi`
run on the same streams (max difference exactly 0.0), and `U - U0 == 0`.

So this run isolates J with no anchoring at all. Comparing it against
lambda = 10 answers "is the benefit from J or from the anchor?":

`lambda_experiment.py --lambda-lasso X` runs any value. Held-out ergodic-variance
ratios (four fresh seeds each, all below 1 in every cell):

| | lambda = 0 | lambda = 0.9 | lambda = 10 |
|---|---|---|---|
| L1-smooth ball | 0.531 (46.9%) at s = 3 | 0.530 (47.0%) at s = 3 | 0.529 (47.1%) at s = 3 |
| unit ball | 0.533 (46.7%) at s = 7 | 0.533 (46.7%) at s = 7 | 0.537 (46.3%) at s = 10 |

**Identical within noise across two orders of magnitude of lambda.** The whole
variance reduction comes from J; the anchor contributes none of it.

How hard the anchor is working, for reference:

| lambda | a bound | a observed | max abs(U - U0) | L | differs from plain Langevin by |
|---|---|---|---|---|---|
| 0 | [1, 1] | [1, 1] | 0 | 894 | 0 (exactly) |
| 0.9 | [0.866, 1] | [0.981, 0.994] | 0.019 | 939 | 2.4e-3 |
| 10 | [0.202, 1] | [0.775, 1] | 0.255 | 1394 | - |

At lambda = 0.9 the anchor is active but barely: `a` stays within 2% of 1. It only
becomes genuinely load-bearing near lambda = 10. That is the expected division of labour - the anchor
exists to handle non-differentiability, not to accelerate - but it is worth
having measured rather than assumed.

Accuracy stays indistinguishable through the useful range of s (|t| <= 1.6 up to
s = 4 on both geometries) and then degrades on the L1 ball exactly as before
(s = 5 gives t = -12.7, s = 7 drives the projection rate to 0.50).

### Running in Colab

The notebooks import `anchored_sgld.py`, which sits next to them in this
repository. Uploading only the `.ipynb` to Colab therefore fails with
`ModuleNotFoundError: No module named 'anchored_sgld'`.

Two fixes:

* **Self-contained versions.** `accuracy_curves_only_colab.ipynb` and
  `accuracy_curves_l1smooth_colab.ipynb` are identical to their counterparts
  except for a first cell that writes `anchored_sgld.py` to the working
  directory with `%%writefile`. Upload one file, run all cells — no upload of
  the module, no network, nothing to `pip install` (Colab already has NumPy,
  SciPy, pandas, scikit-learn and Matplotlib). Both were verified by executing
  them in a directory containing only the notebook.
* **Or upload the module.** Put `anchored_sgld.py` next to the notebook in the
  Colab file browser, or fetch it in a cell, and the original notebooks work
  unchanged.

The `_colab.ipynb` files are generated from the originals, so the embedded
library is the same text as `anchored_sgld.py` and cannot drift from it.

### Accuracy-only notebook

`accuracy_curves_only.ipynb` (and `.executed.ipynb` with outputs) is a minimal
notebook that plots **only training and test accuracy** — one figure, no other
graphs — for the two anchored methods at the configuration found above
(s = 5, unit ball, AR(1) rho_x = 0.99). The supporting paired statistics and the
held-out-seed confirmation are reported as tables rather than plots. Outputs go
to `results_accuracy_only/`.

### L1-smooth ball variant

`accuracy_curves_l1smooth.ipynb` repeats the accuracy-only notebook with the
constraint replaced by the **L1-smooth ball**, the `p = 1` member of the same
smoothed-l^p family: `g(beta) = sum_i sqrt(beta_i^2 + eps^2) <= Lambda`, with
`Lambda = d*eps + R`. The geometry is implemented in `anchored_sgld.py` as
`L1SmoothBallGeometry` and verified to the same standard (skew-symmetry and
divergence exactly 0, `Jn = 0` and `J grad_H = 0` to ~9e-16, projection KKT
residual ~9e-16, radial scaling strictly worse in 20/20 cases, and the
vectorised projection cross-checked against scalar `brentq` at 1.8e-15 and
against SLSQP at 5e-8).

`s = 5` cannot be carried over: `grad_g[i]` saturates towards +/-1 here, so `||J||`
is much larger at the same `s`. The two free constants are instead matched to the
unit-ball configuration on the quantities that drive the dynamics — `R = 1.9`
matches the anchor level (`H(beta_true) = 0.470` vs `0.4725`) and `s = 2.0`
matches the perturbation size (`||alpha J v||/||v|| = 2.38` vs `2.50`).

**Result: non-reversible does NOT beat reversible on this geometry.** Final test
accuracy 0.6958 +/- 0.0094 vs 0.6955 +/- 0.0097; the held-out-seed confirmation
gives pooled −0.00094 (SE 0.00064, t = −1.47), not all batches positive. Four
candidate configurations were selected across `R` in {1.5, 1.9, 3.0} and `s` in
[0.25, 5] and **every one failed held-out confirmation** — the table is in the
notebook. The Euclidean-ball result (pooled t = +4.09) stands unchanged; the
effect is real but small and geometry-dependent.

### Honest scope of this claim

* The win required **changing the design** to an ill-conditioned one. Tuning s
  alone on the specified isotropic problem does not produce an accuracy win, and
  this README does not claim otherwise.
* The effect is small in absolute terms (**+0.24 accuracy points**) and is a
  *convergence-rate* effect, not a better fixed point: both methods target the
  same posterior.
* It is a paired, held-out-confirmed result, not a single lucky configuration —
  but it is one data set, one split, and one constraint set (the unit ball).
* Accuracy still says nothing about posterior fidelity; see the limitations above.
