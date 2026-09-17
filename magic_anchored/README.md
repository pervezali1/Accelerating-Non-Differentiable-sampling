# Constrained non-reversible anchored Langevin on MAGIC Gamma Telescope

A projected, time-changed Langevin sampler for a **non-differentiable** Bayesian
logistic-LASSO posterior on a Euclidean ball, with a state-dependent
skew-symmetric drift that makes the dynamics non-reversible.

```bash
python3 main.py --data ../data/magic04.data --out-dir outputs   # the full run
python3 main.py --quick                                         # pipeline check, <1 min
python3 tests/test_magic_anchored.py                            # 25 unit tests
```

## What is being sampled

The target is the logistic-LASSO potential, a **sum** over the training
observations (not an average -- the tempered alternative exists behind
`--tempered` and is off by default):

```
U(w) = sum_i [softplus(x_i'w) - y_i x_i'w] + beta_0^2/(2 sigma0^2) + lambda sum_{j>=1} |beta_j|
```

The intercept is excluded from the L1 penalty and carries a weak Gaussian prior
instead. `lambda = 0.01 n_train` by default, because the likelihood grows with
`n` and a penalty fixed independently of it would stop mattering.

## The three moving parts

**Anchoring** handles the kink. The smooth anchor `U0` replaces each `|beta_j|`
by `sqrt(beta_j^2 + delta^2)`; the sampler follows `grad U0` but runs its own
clock slow by `a(w) = exp(U - U0)`, which leaves `e^{-U0}/a = e^{-U}` invariant.
Two facts make this cheap and safe, and both are exploited in the code:

* the likelihood and the intercept prior **cancel identically** in `U - U0`, so
  the clock costs `O(p)` and never touches the data. It is computed from the
  penalty directly, never as a difference of two exponentiated potentials;
* `|b| <= sqrt(b^2 + delta^2) <= |b| + delta` termwise gives the a-priori bound
  `-lambda p delta <= log a <= 0`, so the clock never speeds the dynamics up and
  never stalls them. Every recorded value is checked against it.

**The constraint** is a ball around the smooth MAP, `C = {||w - w_center|| <= R}`,
with `R = 1.10 * quantile_{0.999}` of the distance reached by a single
*reversible, unconstrained* pilot chain. The centre and radius are frozen into a
`ConstraintSpec`, written to JSON, and thereafter only read -- so they cannot be
recomputed per `alpha`.

**The skew field** embeds one 3-D hat map per coordinate triple and sums them,
with overlapping cyclic triples `(i, i+1, i+2)` for `i = 0, 2, 4, ...`; at
`D = 11` that is exactly `(0,1,2), (2,3,4), (4,5,6), (6,7,8), (8,9,10), (10,0,1)`.
Three identities survive the sum, and the code verifies all three numerically:

| property | why it holds | why it matters |
|---|---|---|
| `J' = -J` | each block is skew | the perturbation preserves the target |
| `q'J = 0`, `q = w - w_center` | `q'(q x v) = 0` blockwise | the drift is **tangent** to the boundary, so a plain projection never fights it |
| `div J = 0` | `J[i,j]` depends on `q_k` with `k` outside `{i,j}`, so `d J[i,j]/d w_j = 0` *termwise* | **no `div J` correction is added to the drift** |

The update is then
`w_{k+1} = P_C[ w_k - h a_k (I + alpha J_s(w_k)) grad U0(w_k) + sqrt(2 h a_k) xi_k ]`.
Only `alpha * s` matters, so `s = 1` and `alpha` is the knob.

## Two things calibrated, and why that is legitimate

Neither `delta` nor `h` appears in `U`, so calibrating them changes the
algorithm and not the posterior.

`delta` trades off **in both directions at once**, which is why a default value
is a bad idea: the clock falls roughly like `exp(-lambda k delta)` with `k` the
number of coefficients the LASSO has driven onto its kink, while the anchor's
curvature at such a coefficient is `lambda / delta`, so the stable step size
falls as `delta` shrinks. The product `h(delta) * a(delta)` is what actually
advances the dynamics, and it has an interior maximum -- here at
`delta = 3e-3`, where `a ~ 0.41` and the slowest mode relaxes in ~1350
iterations. At `delta = 1e-2` the clock collapses to `a ~ 0.03` and at `3e-4`
the step size does; either end is ~4x slower. The whole table is written to
`outputs/delta_calibration.csv`.

`h = 0.25 / lambda_max(Hess U0(w_center))`. Explicit Euler is stable only for
`h a L < 2` and `a <= 1`, so this stays well inside the stability boundary. The
binding curvature is usually the *penalty's* `lambda/delta`, not the
likelihood's.

## Reading `alpha`: it is not order one

`J_s` is **linear in `q = w - w_center`**, so a posterior that sits a distance
`R` from its own centre sees a perturbation of size `alpha s R / sqrt(m)`. The
rotation matches the gradient at

```
alpha* = sqrt(m) / (s R)  ~  12.6   (R ~ 0.194, m = 6)
```

so at `alpha` of order one the non-reversible drift is **under 3% of the
reversible one** and cannot change anything measurable. That is a statement
about the geometry of this construction, not about non-reversibility, and it is
why the default sweep runs `alpha` up to 200: the specified values
`{0, 0.25, 0.5, 1, 2}` are all in the regime where nothing can happen.
`alpha / alpha*` is the dimensionless strength and is reported in every table.

## Layout

```
config.py     frozen dataclasses: data, target, field, sampler, pilot, experiment, constraint
data.py       load magic04.data, g/h -> 1/0, stratified split, train-only scaler, intercept
target.py     U, U0, grad U0, log a(w), the clock bounds, Hess U0, LogisticTarget
geometry.py   cyclic triples, construct_J, projection, the three identity checks, alpha*
sampler.py    smooth MAP, delta and h calibration, pilot, radius, the projected Euler loop
diagnostics.py ESS / split R-hat (ArviZ, with a fallback), IAT, MSJD, predictive scoring
plots.py      the ten figures, alpha on a single-hue sequential ramp
experiment.py the whole pipeline and the eleven validation checks
main.py       CLI entry point
tests/        25 unit tests; every identity checked against an independent computation
```

## Outputs

`config.json`, `constraint.json`, `validation.json`, `delta_calibration.csv`,
`diagnostics.csv`, `predictive.csv`, `chains.csv`,
`step_size_sensitivity.csv`, `radius_sensitivity.csv`,
`confusion_alpha_*.csv`, `samples/alpha_*.npz` (samples, full trajectories,
clock, radius, drift split, projection flags, seeds), `summary.txt`, and
`figures/` as both 300-dpi PNG and vector PDF.

`main.py` exits non-zero if any validation check fails.

## What is checked automatically

Every stored state inside `C`; `J` skew; `q'J = 0`; the anchoring bounds on
every recorded `log a`; no NaN or infinity; gradient shapes; the intercept
absent from the L1 penalty (verified by *moving* `beta_0` and confirming the
penalty does not change); one shared target, constraint, step size, iteration
count and seed set across every `alpha`; the centre identical bit-for-bit in
every chain; the projection percentage reported. Warnings fire on a projection
frequency above 10%, any R-hat above 1.01, low ESS, unstable parameter norms,
and posterior means that move by more than half a posterior standard deviation
between `h` and `h/2`.

## Caveats

* The scheme is **unadjusted**: there is no Metropolis correction, so the
  invariant law is `O(h)` from the target. That bias is why `h` is conservative
  and swept over `h, h/2, h/4` with the simulated time held fixed.
* One split, one `lambda`, one prior. Predictive scores are reported to confirm
  the rotation does not *bias* the answer, not as evidence that some `alpha`
  predicts better -- these posteriors all target the same law, and differences
  of a few thousandths in test accuracy on 3804 points are inside binomial
  noise. ROC-AUC, log loss and the Brier score are the threshold-free
  comparables.
* The constraint is a modelling choice made for the algorithm's sake. It is
  placed so that it rarely binds; `radius_sensitivity.csv` shows what `0.9R` and
  `1.1R` do.
* `alpha = 0` skips building `J` entirely, as it should, so its ESS-per-second
  advantage in wall-clock terms is real and not an artefact.
