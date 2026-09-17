# Accelerating Non-Differentiable Sampling

**Projected Non-Reversible Anchored Langevin (PNRAL) on the MAGIC Gamma
Telescope binary-classification data set.**

A complete, reproducible implementation of a constrained sampler for a
non-differentiable target: Bayesian logistic regression with a LASSO prior,
sampled on a smoothed $\ell_p$ ball with a state-dependent, skew-symmetric,
divergence-free non-reversible drift built from the gradient of the smoothed
constraint.

---

## 1. The method

### Target

For labels $y_i\in\{0,1\}$ and $z = Xw$ with $w=(\text{intercept},\beta_1,\dots,\beta_{10})$,

$$U(w)=\sum_{i}\big[\operatorname{softplus}(z_i)-y_i z_i\big]
      +\frac{w_0^2}{2\sigma_{\text{intercept}}^2}
      +\lambda_{\text{lasso}}\sum_{j\ge 1}|w_j| .$$

The likelihood is a **sum** over the ~15 000 training rows. The intercept is
never L1-penalised; it carries a weak Gaussian prior.

### Smooth anchor and the anchoring coefficient

$U_0$ replaces $|w_j|$ by $\sqrt{w_j^2+\delta_{\text{anchor}}^2}$, so

$$\log a(w)=U(w)-U_0(w)
  =\lambda_{\text{lasso}}\sum_{j\ge1}\Big(|w_j|-\sqrt{w_j^2+\delta_{\text{anchor}}^2}\Big)
  \in\big[-10\,\lambda_{\text{lasso}}\delta_{\text{anchor}},\,0\big],$$

evaluated **directly** from this formula (never as $\log(e^{U}/e^{U_0})$), so
$a(w)=e^{\log a(w)}\in\big[e^{-10\lambda_{\text{lasso}}\delta_{\text{anchor}}},1\big]$.

The anchored diffusion $\mathrm{d}W=-a(W)\nabla U_0(W)\,\mathrm{d}t+\sqrt{2a(W)}\,\mathrm{d}B$
has stationary density $\pi\propto e^{-U_0}/a=e^{-U}$: substituting
$\pi=e^{-U_0}/a$ into the stationary Fokker–Planck equation gives
$\nabla\cdot\big(-\nabla e^{-U_0}+\nabla e^{-U_0}\big)=0$. No $\nabla a$
correction term is needed.

### Constraint

$$g(w)=\sum_{i=0}^{d-1}\big(w_i^2+\epsilon_{\text{constraint}}^2\big)^{p_{\text{constraint}}/2},
\qquad K=\{w: g(w)\le\Lambda_{\text{constraint}}\},$$

with minimum $g(0)=d\,\epsilon_{\text{constraint}}^{\,p_{\text{constraint}}}$ and slack
$\psi=\Lambda_{\text{constraint}}-g$, $\nabla\psi=-\nabla g$.

> **The constrained posterior is the truncation of the unconstrained posterior**
> $\exp(-U)$ to $K$. Runs at different radii target *different* distributions.

### The non-reversible matrix $J(w)$

For each coordinate triple $(a,b,c)$ with $k=(\nabla\psi_a,\nabla\psi_b,\nabla\psi_c)$,
the cross-product block

$$[k]_\times=\begin{pmatrix}0&-k_3&k_2\\ k_3&0&-k_1\\ -k_2&k_1&0\end{pmatrix}$$

is inserted into rows and columns $(a,b,c)$ of a $d\times d$ zero matrix, and
$J(w)=\texttt{swirl\_scale}\sum_t[k^{(t)}]_\times$. For $d=11$ the primary
(manuscript-faithful) triples are **disjoint**: $(0,1,2),(3,4,5),(6,7,8)$;
coordinates 9 and 10 get reversible drift and diffusion but no direct swirl.

Three properties hold, and are verified numerically at every probe point:

| property | why it holds | measured residual |
|---|---|---|
| $J^\top=-J$ | $[k]_\times$ is skew | `0` (exact) |
| $\operatorname{div}J=0$ | $g$ is separable, so $(\operatorname{div}J)_a=\partial_b\partial_c\psi-\partial_c\partial_b\psi=0$ | `0` (centred differences) |
| $J(w)\,n(w)=0$ on $\{g=\Lambda\}$ | $n\propto\nabla g=-\nabla\psi$ and $[k]_\times k=k\times k=0$ | $\sim10^{-16}$ |

Hence $n^\top J\nabla U_0=0$: the non-reversible current is tangent to the
boundary, and because $\operatorname{div}J=0$ **no $\operatorname{div}J$
correction is added to the drift**.

An **optional overlapping-triple extension**
$(0,1,2),(2,3,4),(4,5,6),(6,7,8),(8,9,10),(10,0,1)$ lets all eleven coordinates
swirl; each block is scaled by the constant
$\texttt{swirl\_scale}/\sqrt{\#\text{triples}}$. It keeps all three properties
and is reported as a labelled sensitivity experiment, not as the main result.
**No state-dependent rescaling of $J$ is ever applied** — that would break
divergence freeness.

### The update

$$w_{k+1}=\Pi_K\Big[w_k-h\,a_k\,(I_d+\alpha J_k)\nabla U_0(w_k)+\sqrt{2ha_k}\,\xi_k\Big],
\qquad \xi_k\sim N(0,I_d),$$

with $\alpha=0$ the constrained **reversible** anchored baseline. $\Pi_K$ is the
exact Euclidean projection, solved from the KKT system: for a scalar multiplier
$\eta>0$ the $d$ equations
$z_i+\eta p z_i(z_i^2+\epsilon^2)^{p/2-1}=y_i$ decouple, each root lies in
$[0,|y_i|]$ with $\operatorname{sign}(z_i)=\operatorname{sign}(y_i)$, and $\eta$
is found by an outer root solve of $g(z(\eta))=\Lambda_{\text{constraint}}$ after
bracketing $\eta_{\text{high}}$ geometrically. For $p_{\text{constraint}}=2$ the
set is a Euclidean ball and an exact radial projection is used instead (never
for $p\neq2$).

The scheme is **unadjusted** (Euler–Maruyama + projection), so it carries the
usual $O(h)$ discretisation bias — which is exactly what the $h$, $h/2$, $h/4$
study measures.

---

## 2. Installation

```bash
pip install -r requirements.txt
```

Then put `data/magic04.data` in place (see `data/README.md`).

## 3. Running

```bash
# full study with the production defaults
python -m pnral.experiment --data-path data/magic04.data --output-dir results

# fast end-to-end smoke test of the code path (synthetic surrogate, not science)
python -m pnral.experiment --synthetic --quick --output-dir /tmp/pnral_smoke

# fix the threshold and skip the pilot procedure
python -m pnral.experiment --lambda-constraint 4.25 --step-size 5e-5

# the test suite (64 tests, ~25 s)
python -m pytest
```

The notebook `notebooks/magic_pnral_experiment.ipynb` runs the same study
step by step — data, anchor, constraint, $J$ verification, projection demo,
the full comparison, all tables and all figures inline.

### Useful flags

`--alphas 0,0.1,0.25,0.5,1.0` · `--chains` · `--iterations` · `--burn-in` ·
`--thinning` · `--step-size` · `--lambda-lasso` · `--sigma-intercept` ·
`--delta-anchor` · `--p-constraint` · `--epsilon-constraint` ·
`--lambda-constraint` · `--swirl-scale` · `--triples-mode {disjoint,overlapping}` ·
`--no-sensitivity` · `--quick`

## 4. Choosing $\Lambda_{\text{constraint}}$

It is never picked arbitrarily. The default procedure is:

1. smooth MAP of $U_0$ by L-BFGS-B;
2. an **unconstrained** reversible ($\alpha=0$) anchored pilot chain at a
   conservative step size ($0.25/L$);
3. $q_{0.999}$ = empirical 0.999 quantile of $g(w_k)$ over the retained pilot
   draws, $g_{\min}=d\epsilon^p$;
4. $\Lambda_{\text{constraint}}=g_{\min}+1.10\,(q_{0.999}-g_{\min})$;
5. **frozen** before any final chain and reused for *every* $\alpha$;
6. the fraction of pilot draws with $g>\Lambda_{\text{constraint}}$ is reported;
7. a sensitivity sweep over
   $g_{\min}+0.90\,(\Lambda-g_{\min})$, $\Lambda$, $g_{\min}+1.10\,(\Lambda-g_{\min})$.

`--lambda-constraint` bypasses the pilot entirely.

## 5. Step size

The smoothness guide $L_{\text{likelihood}}\le0.25\,\|X\|_2^2$ (from
$p(1-p)\le1/4$) supplies the starting point $h_0=0.5/L$; $h$ is then halved
until a trial chain **at the largest $\alpha$** is finite, does not drift away
from the MAP and projects on at most 25 % of its iterations. The accepted $h$ is
shared by every $\alpha$. Because the likelihood sums over ~15 000 rows, the
stable $h$ is small (order $10^{-5}$) — expected, not a bug.

## 6. Layout

```
pnral/
  config.py                 all parameters, JSON round-trip
  data.py                   load / encode / stratified split / scale / intercept
  target.py                 the non-smooth potential U
  anchor.py                 U0, grad U0, log a, a and their bounds
  constraint.py             g, grad g, psi, the pilot rule for Lambda
  nonreversible_matrix.py   J(w), triples, skew / divergence / tangency checks
  projection.py             exact Euclidean projection onto K + KKT residuals
  sampler.py                the PNRAL update, MAP, initialisation, calibration
  diagnostics.py            ArviZ ESS / R-hat, the 14 checks, figures 1-11
  prediction.py             posterior-averaged predictions, figures 12-14
  experiment.py             the end-to-end driver and the CLI
tests/                      64 tests covering every automatic check
notebooks/                  the full experiment as a notebook
```

## 7. Outputs

```
results/
  config.json                    the complete configuration
  constraint.json                threshold, pilot statistics, triples
  preprocessing.json             scaler parameters, label encoding, shapes
  standard_scaler.joblib         the fitted scaler
  run_metadata.json              runtime, versions, step-size calibration trace
  summary.md                     the human-readable record of the run
  samples/*.npz                  retained draws + every per-iteration trace
  tables/automatic_checks.csv    the fourteen checks
  tables/diagnostics_summary.csv per-alpha diagnostics
  tables/coefficient_summary.csv per-coefficient posterior + ESS + R-hat
  tables/predictive_results.csv  test-split metrics
  tables/diagnostics_traces.csv  thinned per-iteration traces
  figures/fig01..fig16           each as PDF and 300-dpi PNG
```

Recorded at **every iteration**: $w_k$, $U$, $U_0$, $\log a$, $a$, $g$,
$\Lambda-g$, $\|\nabla U_0\|$, reversible / non-reversible / total drift norms,
$\|J\|_2$, whether a projection occurred, the projection distance and elapsed
time.

Reported per $\alpha$: ESS per coefficient, min/median ESS, ESS per second,
split $\hat R$, integrated autocorrelation time, mean squared jumping distance,
projection frequency and mean distance, posterior means and standard
deviations, runtime and gradient-evaluation count.

Warnings fire when any $\hat R>1.01$, projection frequency $>10\%$, a state
violates the constraint, ESS is very low, $\|J\|$ grows numerically large, or
the $h$ vs $h/2$ or the radius sweeps move a posterior mean by more than
0.25 posterior standard deviations.

## 8. Automatic checks

The driver runs all fourteen and writes them to `tables/automatic_checks.csv`:

1. $g(0)=d\,\epsilon^p$ · 2. $\Lambda>g(0)$ · 3. every sampled state satisfies
$g(w_k)\le\Lambda+\text{tol}$ · 4. $J^\top=-J$ · 5. finite-difference
$\operatorname{div}J\approx0$ · 6. $J n\approx0$ on the boundary · 7. the anchor
bounds · 8. the intercept is excluded from the L1 penalty · 9. the projection
satisfies the KKT conditions · 10. $\alpha=0$ removes *only* the non-reversible
drift (verified bit-for-bit against a hand-rolled reversible update) ·
11. no state-dependent rescaling of $J$ · 12. the same $\Lambda$ for every
$\alpha$ · 13. no NaN or infinite quantities · 14. labels encoded `g -> 1`,
`h -> 0`.

## 9. Evaluation discipline

Predictions use posterior-averaged probabilities
$\bar p_i=\frac1M\sum_m\operatorname{expit}(x_i^\top w^{(m)})$, and the report
covers accuracy, balanced accuracy, ROC-AUC, log loss, Brier score, the
confusion matrix, sensitivity, specificity, precision, recall, F1 and a
calibration curve. MAGIC is imbalanced (~65 % gamma), so balanced accuracy,
ROC-AUC, sensitivity, specificity and calibration are the metrics to read first.

**$\alpha$ is selected from training diagnostics only** (largest minimum ESS per
second). The test split is touched exactly once, for the final evaluation.

## 10. Defaults

| parameter | default | note |
|---|---|---|
| `lambda_lasso` | 100.0 | the likelihood sums over ~15 000 rows, so an $O(10^2)$ penalty is needed to move the posterior by a noticeable fraction of a posterior SD |
| `sigma_intercept` | 10.0 | weak Gaussian prior, intercept only |
| `delta_anchor` | 0.01 | with `lambda_lasso=100` keeps $a$ in a numerically comfortable range while staying genuinely state dependent |
| `p_constraint`, `epsilon_constraint` | 1.5, 0.05 | as specified |
| `swirl_scale` | 1.0 | primary experiment |
| chains / iterations / burn-in / thinning | 4 / 20 000 / 5 000 / 1 | all configurable |
| `alphas` | 0, 0.1, 0.25, 0.5, 1.0 | common random numbers across $\alpha$ |

All seeds are fixed and recorded (`split_seed`, `pilot_seed`, `init_seed`,
`chain_seed_base`); everything is `float64`.
