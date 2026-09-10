# Accelerating Non-Differentiable Sampling

Does an irreversible (skew-symmetric) perturbation speed up a **derivative-free**
MCMC sampler?  This repository answers that question on four binary
classification posteriors by measuring how fast test-set accuracy converges from
a cold `w = 0` start, for four choices of the perturbation:

| variant | field | divergence correction |
|---|---|---|
| `zero` | `J = 0` (reversible baseline) | not needed |
| `constant` | `J_a = alpha A`, `A` skew-symmetric and constant | zero by construction |
| `state` | `J_s(w) = alpha s(w) A`, radial profile `s` | included |
| `state_nocorr` | the same `J_s(w)` | **dropped** |

![accuracy from the w = 0 start](figures/accuracy_four_datasets.png)

## Provenance

The repository held nothing but a one-line README when this code was written, so
the sampler here is a **reconstruction** built to match the earlier two-panel
figure (its legend, its `w = 0` start, its 600-iteration axis, its 5-95% walker
band and its exact-posterior reference line), not the original implementation.
Everything it assumes is stated in [Method](#method) and is a knob in the code;
swap in your own kernel by implementing `nds.skew.SkewField` or by replacing
`nds.target.LogisticPosterior`.  Breast Cancer Wisconsin and Spambase are the two
datasets added here, alongside Titanic and MAGIC.

## Method

### Target

Bayesian logistic regression: `U(w) = -log p(y | X, w) - log p(w)` with a
Gaussian prior (`prior_scale = 2`, the intercept effectively unpenalised at
scale 10).  Features are standardised on the training split and carry an
intercept column, and the posterior is sampled on a stratified 70% training
split while accuracy is measured on the held-out 30%.

### The sampler is derivative free

The dynamics are the irreversible Langevin diffusion of Ma, Chen and Fox (2015),

```
dw = [-(D + J(w)) grad U(w) + Gamma(w)] dt + sqrt(2 D) dB,
Gamma_i(w) = sum_j d/dw_j (D_ij + J_ij(w)),
```

discretised by Euler-Maruyama and corrected by Metropolis-Hastings.  `grad U` is
never evaluated: the drift uses the central-difference surrogate

```
ghat_j(w) = [U(w + eps e_j) - U(w - eps e_j)] / (2 eps),      eps = 1e-2,
```

which costs `2 d` potential evaluations per step and touches only the linear
predictor, `X (w + eps e_j) = Xw + eps X_j`, so no matrix product is repeated.
`Gamma` is the divergence of the field *the user chose*, available in closed
form, so it needs no derivative of `U` either.  (The analytic gradient in
`nds.target.LogisticPosterior.grad` is used only to build the reference
posterior, never by a sampler under test.)

### Why the Metropolis step matters for the comparison

Because `mu(w) = w + h[-(D + J(w)) ghat(w) + Gamma(w)]` is a deterministic
function of `w`, the accept/reject step makes `exp(-U)` the **exact** invariant
law for every variant in the table -- including the one that drops `Gamma`.
Dropping the correction is therefore an efficiency question, not a bias
question, and the `state` and `state_nocorr` curves in the accuracy figure lie
on top of each other by construction.  `tests/test_nds.py` asserts this: all
four variants agree with a long reference run to within half a posterior
standard deviation.

The correction earns its keep in the *uncorrected* diffusion, which is what
`experiments/run_correction_bias.py` measures -- see
[the correction term](#the-correction-term-only-matters-without-the-metropolis-step)
below.

### The fields

`A` is a random skew-symmetric matrix (fixed seed) normalised to unit spectral
norm and then expressed in the geometry of the preconditioner, `A <- L A L^T`
with `D = L L^T`, which keeps it skew and puts `alpha = 1` at "a rotation as
strong as the reversible part" in whitened coordinates.  The state-dependent
field modulates it radially,

```
J_s(w) = alpha s(w) A,   s(w) = 1 - exp(-r(w)^2 / 2 rho^2),   r(w)^2 = w^T D^{-1} w,
Gamma(w) = alpha A grad s(w),   grad s(w) = (D^{-1} w / rho^2) exp(-r(w)^2 / 2 rho^2),
```

so the rotation is switched off at the `w = 0` start, where `|grad U|` is
largest, and switched on once the walker has travelled a whitened distance of
order `rho`.

### Warm-up, preconditioner, step size

Per dataset, two `J = 0` warm-up rounds (derivative free, no held-out data)
produce the shrunk sample covariance used as `D` and the radial scale
`rho = half the whitened distance of the pilot ensemble from the origin`.  Each
variant then gets **its own** step size, bisected so that a run started at
`w = 0` accepts about half of its proposals over the whole transient.  Tuning on
the transient rather than at stationarity is deliberate: a step size that is
only good near the mode leaves a Metropolis chain frozen at the start.  Every
variant is tuned by the same procedure, and the calibrated values are reported
in `results/summary.csv`.

### Reference posterior

The dashed line in each panel is the posterior predictive accuracy of a long
preconditioned MALA run that *does* use analytic gradients and Hessians (MAP,
Laplace preconditioner, 8 chains, 20k iterations after warm-up, thinned).  Its
two halves agree to four decimals on every dataset.

### Accuracy metric

At iteration `t` each walker predicts with the running posterior predictive
mean, `mean_{s<=t} sigmoid(X_test w_s)`, thresholded at one half with ties
counting as one half -- so the `w = 0` start scores exactly 0.5.  Panels show
the mean over 32 walkers, with the 5-95% band across walkers for the baseline.

## Datasets

| dataset | n (train/test) | d | source |
|---|---|---|---|
| Titanic (survival) | 623 / 268 | 10 | Kaggle Titanic training split, via the `seaborn-data` mirror |
| MAGIC Gamma Telescope | 13314 / 5706 | 11 | UCI `magic04.data` (19020 events) |
| Breast Cancer Wisconsin | 398 / 171 | 31 | shipped with scikit-learn (`load_breast_cancer`) |
| Spambase | 3221 / 1380 | 58 | UCI `spambase.data`, features `log1p`-scaled |

`nds/data.py` caches the raw files under `data/` and appends their SHA256 digest
to `data/checksums.txt` on download.  UCI and OpenML are unreachable from some
environments, including the one this was run in, so the two UCI files are fetched
from a public GitHub mirror; the digests make a mirror change visible.

## Reproducing

```bash
pip install -r requirements.txt
python3 tests/test_nds.py                 # correctness checks, ~2 minutes
./experiments/run_all.sh                  # four datasets, one process each
python3 experiments/plot_accuracy.py      # figures + results/summary.{csv,md}
python3 experiments/run_alpha_sweep.py && python3 experiments/plot_alpha_sweep.py
python3 experiments/run_correction_bias.py && python3 experiments/plot_correction_bias.py
```

Single runs are configurable, for instance

```bash
python3 experiments/run_accuracy.py --datasets magic --alpha 0.5 --geometry identity
```

## Layout

```
nds/data.py        dataset loaders, caching, standardisation, splits
nds/target.py      logistic posterior, central-difference surrogate, prediction
nds/skew.py        J = 0, constant, state-dependent fields and their divergence
nds/sampler.py     Metropolis-corrected irreversible steps, warm-up, calibration
nds/reference.py   MAP, Laplace covariance, long gradient-based reference chain
nds/metrics.py     autocorrelation, ESS, iterations-to-reference
experiments/       runners and plotting scripts
results/           traces (.npz), per-dataset summaries, summary.csv
figures/           the figures referenced above
```
