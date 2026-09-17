# Non-reversible anchored Langevin with block state-dependent skew-symmetric matrices

Constrained Bayesian logistic regression on **MAGIC Gamma Telescope** and **Titanic**, comparing
four projected stochastic-gradient samplers that differ only in $(\rho,\alpha)$:

| Method | $\rho$ | $\alpha$ | Style in every figure |
|---|---:|---:|---|
| Projected SGLD | $0$ | $0$ | gray dashed |
| Non-reversible SGLD | $0$ | $1$ | orange dash-dot |
| Reversible anchored Langevin | $\log 2$ | $0$ | blue solid |
| Non-reversible anchored Langevin | $\log 2$ | $1$ | green solid |

Common update (one implementation, four settings):

$$\beta_{k+1}=\Pi_K\!\left[\beta_k-h\,a_k\big(v_k+\alpha J(\beta_k)v_k\big)+\sqrt{2ha_k}\,\xi_k\right],
\quad v_k=\widehat G_k+\rho\nabla H_K(\beta_k),\quad a_k=e^{-\rho H_K(\beta_k)} .$$

## Files

| File | What it is |
|---|---|
| `nonreversible_anchored_langevin.ipynb` | The notebook, **executed**, with all outputs. 13 sections: configuration, data, preprocessing, posterior, constraints/projections, anchor, block matrices, implementation checks, sampler, experiments, figures, step-size sensitivity, interpretation. |
| `nral.py` | Reusable library: geometries, projections, anchor, block matrices, gradients, `run_sampler`, `run_experiment`, metrics, figure helpers. Importing it has no side effects. |
| `notebook_source.py` | Single source of truth (jupytext "percent" format). Cells tagged `lib` are exported to `nral.py`; cells tagged `run` are notebook drivers. |
| `make_notebook.py` | Builds `nonreversible_anchored_langevin.ipynb` **and** `nral.py` from `notebook_source.py`, so the two can never drift apart. |
| `fetch_data.py` | Downloads the two datasets into `data/`, official sources first, and validates them against the official schema. |
| `figures/` | 4 required figures + combined figure + 4 step-size-sensitivity figures, each as 300-dpi PNG and vector PDF. |
| `results/` | Per-run accuracies, coefficient checkpoints, diagnostics, summaries, split indices, preprocessing parameters, seeds, package versions. Figures are regenerable from these alone. |

## Reproducing

```bash
pip install -r requirements.txt
python fetch_data.py                 # writes data/magic04.data and data/titanic_train.csv
python make_notebook.py              # rebuild the notebook and nral.py from the source
NRAL_MODE=full jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 nonreversible_anchored_langevin.ipynb
```

`NRAL_MODE=quick` runs a 5-replicate / 200-iteration smoke test. **Quick-mode output is not the
experiment** and the notebook labels it as such wherever it appears.

## The four experiments

$d=9$ coefficients, no intercept, $m=30$, stratified 80/20 split (seed 2027), sampler seed 3000,
$R=100$ replicates, $h=10^{-4}$, checkpoints at iteration 0 and every 10 iterations.

| Experiment | Constraint | $s$ | Iterations | train / test |
|---|---|---|---:|---|
| MAGIC ball | $\|\beta\|^2\le2$ | $(5,5,5)$ | 1000 | 15216 / 3804 |
| MAGIC smoothed $\ell_p$ | $p=2.4,\ \varepsilon=0.2,\ \Lambda=4$ | $(5,5,5)$ | 1000 | 15216 / 3804 |
| Titanic ball | $\|\beta\|^2\le2$ | $(2,7,2)$ | 1500 | 712 / 179 |
| Titanic smoothed $\ell_p$ | $p=2.4,\ \varepsilon=0.18,\ \Lambda=4$ | $(2,7,2)$ | 2000 | 712 / 179 |

## Declared design choices

These are **ours**, not verified reconstructions of the source:

1. **MAGIC preprocessing** -- train-only `StandardScaler` then train-only
   `PCA(n_components=9, whiten=False, svd_solver="full")`; the constraint applies to the PCA
   coefficients.
2. **Titanic preprocessing** -- nine features in a fixed order (Age, SibSp, Parch, Fare, female,
   Pclass2, Pclass3, EmbarkedQ, EmbarkedS) with train-median imputation, train-fitted
   standardisation of the four numeric columns, references Pclass 1 / embarkation C, train-mode
   imputation of `Embarked`, no rows dropped for missing `Age`.
3. **"100 samples" $= R = 100$ independent sampler replicates** on one fixed dataset and split.
   The shaded bands are repeat-run variability conditional on that split: **not** confidence
   intervals and **not** posterior credible intervals.
4. **The bounded anchor** $U_0=U+\rho H_K$, $\rho=\log2$, so $a\in[\tfrac12,1]$ on $K$. It is an
   algorithmic reference potential: it does not change the target and has no guaranteed speed
   advantage.
5. **$h=10^{-4}$ is the reported candidate**, not a verified accurate step. Section 12 measures
   how much it can be trusted instead of assuming it.

## What the run found

*No winner is presupposed.* The measured outcome at $h=10^{-4}$:

* **Titanic** is in a sane step-size regime (median drift step $\approx0.02$--$0.055$ against a
  set diameter $\approx2.8$--$4.1$). All four methods land near 0.75--0.81 test accuracy; on the
  ball the four are within about one repeat-run SD of each other.
* **MAGIC is not.** Because the potential is the **sum** over 15216 rows,
  $\|\nabla U\|\approx6\times10^3$ and a single drift step is $O(0.6)$ for $\alpha=0$ and
  $O(1.2)$--$O(1.9)$ for $\alpha=1$, against a set diameter of $2.83$ (ball) / $4.07$
  (smoothed). The non-reversible arms are projected on $\approx99\%$ of iterations and sit near
  0.51--0.57 test accuracy while the reversible arms reach $\approx0.77$.
* Refining to $h/2$ and $h/4$ at **fixed simulated time** $t=kh$ gives an experiment-specific
  answer, reported as measured:
  * MAGIC-ball and Titanic-$\ell_p$: the non-reversible arms move monotonically back toward the
    reversible ones on every observable -- non-reversible anchored on MAGIC-ball goes
    0.561 → 0.594 → 0.708 test accuracy with $U$ = 13742 → 12188 → 8980, and on Titanic-$\ell_p$
    0.763 → 0.779 → 0.804 with $U$ = 390 → 367 → 349. That is a **discretisation artefact** at
    $h=10^{-4}$, not a property of the continuous limit.
  * Titanic-ball: all four arms are comparable at $h$, $h/2$ and $h/4$ alike.
  * MAGIC-$\ell_p$: $U$ (20116 → 18945 → 16674) and the projection rate move in the same
    direction, but **accuracy has not recovered by $h/4$** (0.500 → 0.482 → 0.530). Even $h/4$
    leaves the non-reversible drift step near 0.47 against a set of diameter 4.07. We report this
    and do **not** extrapolate to where that arm would settle at a genuinely small step.
* A bounded, projected chain is not an accurate one: the projection keeps $\beta_k\in K$ no
  matter how large the step is, which is why projection rates are reported alongside every
  result.

Classification accuracy measures prediction. It does not by itself demonstrate posterior
convergence, correct uncertainty, faster mixing, or reduced asymptotic variance. See
Section 13.3 of the notebook.

## Implementation checks (Section 8, all pass at machine precision)

* $\nabla U$ and $\nabla H_K$ against central finite differences.
* The mini-batch constant $n_{\rm train}/m$ by **enumerating every** $\binom{7}{3}$ batch on a toy
  problem.
* $J^\top=-J$, $\operatorname{div}J=0$, $J\,n=0$ on $\partial K$, $J\nabla H_K=0$, and
  matrix-free block cross products against the explicit $9\times9$ matrix.
* That the **ball** matrix *fails* $Jn=0$ on the smoothed set -- the reason the smoothed-set
  matrix is built from $\nabla g$ rather than $\beta$.
* $D=\Lambda-9\varepsilon^p>0$; $H_K\in[0,1]$ and $a\in[\tfrac12,1]$ on $K$; the unit-ball
  initialisation is feasible for every $K$ used here.
* Projection feasibility, KKT residuals, agreement with a `brentq` scalar reference over
  proposal scales from just-outside to far-outside, and that radial scaling is **not** the
  Euclidean projection at $p=2.4$.
* $J\nabla U_0\neq0$ at visited states.

## Data

The notebook and `fetch_data.py` prefer the official sources:

* MAGIC -- <https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope> (or `ucimlrepo`)
* Titanic -- <https://www.kaggle.com/competitions/titanic/data> (labelled `train.csv`; the
  competition's unlabelled `test.csv` is never used as an evaluation set)

**In the environment where this was executed both official endpoints are blocked by the outbound
network policy**, so the files were taken from public mirrors and then validated against the
official specification before use: MAGIC 19020 rows, 10 predictors, class counts g = 12332 /
h = 6688, first and last records matching `magic04.data`; Titanic 891 rows, the 12 competition
columns, 549/342 survived, 177 missing `Age`, 2 missing `Embarked`. SHA-256 digests of the exact
bytes used are recorded in `results/run_config.json`. Raw data are not committed; run
`python fetch_data.py`.

---

# Follow-up: searching the block strength `s` (anchored arms only)

Question: **is there a block strength at which non-reversible anchored Langevin beats the
reversible anchored arm?** Scripts: `strength_search.py` (sweep over `s`), `hs_search.py`
(joint sweep over `h` and `s`), `final_s5.py` (the requested `s = 5`, with an independent
confirmation run). Outputs in `figures/strength/` and `results/strength/`.

## Protocol

* Two arms only, both with `rho = log 2`: reversible (`alpha = 0`) and non-reversible
  (`alpha = 1`). The reversible arm does not depend on `s`, so it is run once per cell and
  reused as the baseline.
* Every comparison is **paired** — same initial states, same mini-batches, same Gaussian
  increments — so the paired standard error is far smaller than the run-to-run SD.
* `s = 0` is a control: with `J = 0` the two arms come out bit-identical (paired difference
  exactly 0), which verifies the pairing.
* **Selection uses training data only** (paired training gain, or mean training accuracy over
  the last 25% of checkpoints). Test labels select nothing.
* Because the selected cell is a maximum over a grid, and the maximum of noisy estimates is
  biased upward, the reported numbers and figures come from a **confirmation run on independent
  replicates** (sampler seed 4100 vs the 3000 used to search).

## Answer

**At the originally specified `h = 1e-4`, no `s` wins.** Across 14 strength triples the best
paired training gains were `+0.0000` to `+0.0008` with standard errors of the same size, and
`s = 5` loses heavily on MAGIC (`-0.190` ball, `-0.254` smoothed) because one non-reversible
drift step there exceeds the diameter of `K`.

**At `h = 1e-5`, `s = 5` wins on Titanic** — confirmed on independent replicates, `R = 100`:

| | reversible train | non-rev train | paired Δ train | reversible test | non-rev test | paired Δ test |
|---|---|---|---|---|---|---|
| Titanic ball | 0.7547 ± 0.0243 | **0.7675 ± 0.0221** | **+0.0127 ± 0.0029** (t = 4.3) | 0.7660 ± 0.0330 | **0.7854 ± 0.0268** | **+0.0193 ± 0.0036** (t = 5.4) |
| Titanic smoothed ℓ_p | 0.7754 ± 0.0171 | **0.7823 ± 0.0163** | **+0.0069 ± 0.0021** (t = 3.2) | 0.7893 ± 0.0231 | **0.7950 ± 0.0210** | +0.0057 ± 0.0033 (t = 1.7) |

Full training loss `U` agrees: 387.1 → 383.7 (ball) and 368.8 → 365.0 (smoothed).

**On MAGIC no step size produced a win.** The gain rises towards zero as `h` shrinks, but even
at the most favourable cell it stays inside the noise — ball `h = 3e-7`: `+0.0024 ± 0.0014`
(t = 1.8); smoothed `h = 1e-7`: `+0.0040 ± 0.0041` (t = 1.0) — and those cells cost a lot of
absolute accuracy (0.749 and 0.646 training, against 0.781 at `h = 1e-5`).

## What the win is, and what it is not

It is a **convergence-rate** win at a fixed iteration budget, which is what a non-reversible
drift is supposed to deliver. At `h = 1e-5` the Titanic chains are still inside their transient
after 1500 / 2000 iterations, and the rotation gets them further along it. Run them to
convergence (`h = 1e-4`, same budget) and both arms land at the same place and the gap vanishes.
So the correct claim is "reaches a given accuracy in fewer iterations at this step size", **not**
"converges to a better posterior". The figures plot the required common `[0, 1]` accuracy axis;
an inset zooms the same curves so the separation is legible.

`h` was chosen per experiment from a small grid on a training-only criterion, which is a real
selection cost and is disclosed here and in every figure caption.
