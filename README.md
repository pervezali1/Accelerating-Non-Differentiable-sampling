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

## Follow-up: what does the smoothed-set level `Λ` do?

`lambda_study.py`; table in `results/strength/lambda_study.csv` and
`results/strength/lambda_spec_init.csv`.

Geometry first (`p = 2.4`, `ε = 0.2 / 0.18`, `d = 9`):

| Λ | D = Λ − 9ε^p | radius of K (axis → diagonal) | ‖∇g‖ on ∂K | unit-ball init feasible? |
|---|---|---|---|---|
| 4 | 3.81 / 3.85 | 1.74 → 2.05 | 4.59 | yes |
| 2 | 1.81 / 1.85 | 1.27 → 1.49 | 2.96 | yes |
| **1** | **0.81 / 0.85** | **0.90 → 1.04** | **1.84** | **no** (max g on the unit ball is 1.216 / 1.170) |
| 0.5 | 0.31 / 0.35 | 0.60 → 0.67 | 1.05 | no |

`Λ = 1` keeps `D > 0`, so the anchor and the bound `1/2 ≤ a ≤ 1` are untouched. Its costs:

1. **It invalidates the specified initialisation** — the unit ball no longer fits inside `K`, so
   `beta0` must be shrunk or projected. The study uses a common radius 0.55.
2. **Titanic test accuracy falls** 0.807 → 0.785 (`h = 1e-4`, reversible arm). MAGIC barely moves
   (0.770 → 0.768): accuracy is invariant under `β → cβ`, so a tighter Λ bites only through the
   anisotropy of the ℓ_p set distorting the *direction*, and Titanic's MLE (‖β‖ = 3.59) is
   distorted far more than MAGIC's (2.17).
3. **The chain gets pinned to the boundary** — MAGIC reversible projection rate at `h = 1e-5`
   goes 5% (Λ = 4) → 79% (Λ = 1) → 93% (Λ = 0.5), i.e. more projection bias.

Its one real benefit: since `J`'s axis is `−s ∇g` and `‖∇g‖` falls 2.5×, `Λ = 1` makes `s = 5`
much less destructive on MAGIC — the paired gap at `h = 1e-5` goes −0.208 (Λ = 4) → −0.082
(Λ = 1) → −0.033 (Λ = 0.5). It never becomes a win, and lowering `s` achieves the same thing
without touching the model.

With the **specified** unit-ball init (`h = 1e-5`, `s = 5`, `R = 100`) the Titanic win is
strongest at the original Λ: `Λ = 4` gives `+0.0069 ± 0.0021` train (t = 3.2), `Λ = 2` gives
`+0.0049 ± 0.0023` (t = 2.2), and `Λ = 1` cannot run that init at all.

**Λ is part of the model** (`π_K ∝ e^{−U} 1_K`), not a sampler knob — changing it changes the
posterior being targeted. `s` is purely algorithmic and leaves the target alone. They overlap in
what they do to `‖J‖`, so `s` is the right knob for that job.

## Follow-up: `s = 0.25` and `s = 1` on MAGIC

`final_s5.py` is parameterised by `--s`, `--experiments` and `--h-scan`; outputs in
`results/strength/s*_step_scan.csv` and `s*_confirmation.json`.

How much of the drift `J` actually contributes on MAGIC (median `‖J∇U₀‖ / ‖∇U₀‖`):

| s | ball | smoothed ℓ_p |
|---|---|---|
| 0.25 | 0.084 | 0.140 |
| 1 | 0.335 | 0.559 |
| 5 | 1.674 | 2.795 |

**`s = 0.25` is safe but inert.** It never does damage — the ball is untouched at every step size
and the ℓ_p set loses only −0.003 at `h = 1e-4` — but it produces no gain either. Confirmed at
`R = 200` on independent replicates: `+0.0000 ± 0.0002` (ball, t = 0.3) and `+0.0000 ± 0.0001`
(ℓ_p, t = 0.4) on training. At `s = 0.25` the rotation is under a fifth of the drift, which is
too weak to change the trajectory.

Note: the `R = 25` sweep had shown `+0.00049 ± 0.00019` (t = 2.6) for `s = 0.25` on the ℓ_p set at
`h = 1e-6`. **That did not replicate** — it was the maximum of a grid of noisy tiny estimates.
This is exactly what the independent-confirmation step exists to catch.

**`s = 1` does win on MAGIC**, at `h = 3e-7`, `R = 200`, independent replicates:

| | reversible | non-reversible | paired Δ |
|---|---|---|---|
| ball — train | 0.7478 ± 0.0163 | 0.7491 ± 0.0159 | +0.0013 ± 0.0005 (t = 2.4) |
| ball — test | 0.7445 ± 0.0171 | 0.7461 ± 0.0162 | +0.0016 ± 0.0006 (t = 2.8) |
| ℓ_p — train | 0.7552 ± 0.0134 | 0.7583 ± 0.0125 | +0.0032 ± 0.0007 (t = 4.6) |
| ℓ_p — test | 0.7518 ± 0.0139 | 0.7556 ± 0.0133 | +0.0038 ± 0.0008 (t = 4.9) |

Training loss agrees (8367.5 → 8335.8 on the ℓ_p set). Caveat: the scan at `R = 60` put the ℓ_p
point estimate at `+0.0007 ± 0.0011`, five times smaller — the effect is small relative to
run-to-run noise, and the `R = 200` confirmation is the number to trust, not the scan.

**The same caveat as Titanic applies.** At `h = 3e-7` absolute accuracy is 0.745–0.756, well below
the 0.774 reachable at larger `h`: the win lives in the transient, where the metric still has
headroom. Run MAGIC to convergence (`h = 1e-5`) and the reversible arm reaches 0.781 train /
0.774 test against an unconstrained-MLE ceiling of 0.781 / 0.776 — about 0.002 of room left, so
no sampler can demonstrate more than that on accuracy at that step size.

**Summary of the strength sweet spot:** `s = 5` is far too large for MAGIC (rotation 1.7–2.8×
the drift), `s = 0.25` far too small (under 0.2×), `s = 1` is where it works (0.3–0.6×).

---

# Exact-gradient anchored Langevin with an ℓ₁ (LASSO) potential

`exact_anchored.py`; figures in `figures/exact/`, sweep and confirmation in `results/exact/`.

This is the construction the method is actually for: the anchor now smooths a **genuinely
non-differentiable** potential rather than acting as a constraint barrier.

```
U(w)   = Σ_j [softplus(x_j·w) − y_j x_j·w] + w₀²/(2σ²) + λ Σ_{j≥1} |w_j|          (non-smooth)
U₀(w)  = Σ_j [softplus(x_j·w) − y_j x_j·w] + w₀²/(2σ²) + λ Σ_{j≥1} √(w_j²+δ²)      (smooth)
a(w)   = exp(U − U₀) = exp(−λ Σ_{j≥1} [√(w_j²+δ²) − |w_j|])

w_{k+1} = Π_K[ w_k − η a(w_k) ∇U₀(w_k) + η α a(w_k) J_s(w_k) ∇U₀(w_k) + √(2η a(w_k)) ξ_{k+1} ]
```

The **exact** gradient is used throughout — no mini-batching anywhere. Note the `J` term now
carries a **plus** sign (as specified); `J` is skew, so this simply rotates the opposite way from
the earlier runs.

## Declared choices (the spec fixes the form, not these numbers)

* **d = 10**: intercept `w₀` plus the nine features. The Gaussian acts on `w₀`, the LASSO on
  `w₁..w₉` — exactly the 3×3 block structure `J` needs. `J`'s intercept row and column are zero.
* **K constrains the full vector** (`‖w‖² ≤ 2`, or `g(w) = Σ_{i=0}^{9}(w_i²+ε²)^{p/2} ≤ Λ`).
  Leaving `w₀` free would make `K` non-compact and break the reflection argument. `g_min` is now
  `10ε^p`, so `D = Λ − 10ε^p` = 3.790 / 3.839 > 0.
* `σ = 10`; `λ_lasso = 0.01 · n_train` (152.2 MAGIC, 7.12 Titanic) so the penalty keeps a fixed
  weight against a *summed* log-likelihood; `δ = log2/(9λ)` so that **a ∈ [½, 1] everywhere** —
  the same bound the earlier `ρ = log2` anchor had.

## Verification

`a = exp(U−U₀)` to 1e-12; `a ∈ [0.500000, 0.999947]` over a wide sample; exact `∇U₀` matches
central differences to 3e-10; `J' = −J` and `div J = 0` exactly, `‖Jn‖/‖n‖ ≤ 5e-16` on both
boundaries, matrix-free equals explicit to 5e-16, intercept row/column identically zero;
projection infeasibility ≤ 3e-15; unit-ball init feasible for both geometries.

## Results (confirmation on independent replicates, seed 4100, R = 150)

`η` and `s` selected by a **training-only** sweep over 5 step sizes × 4 strengths.

| | η | s | reversible train | non-rev train | paired Δ train | reversible test | non-rev test | paired Δ test |
|---|---|---|---|---|---|---|---|---|
| **Titanic ball** | 1e-5 | 5 | 0.7630 ± 0.0264 | **0.7763 ± 0.0216** | **+0.0133 ± 0.0020** (t = 6.6) | 0.7690 ± 0.0360 | **0.7880 ± 0.0262** | **+0.0190 ± 0.0029** (t = 6.5) |
| **Titanic ℓ_p** | 1e-5 | 5 | 0.7822 ± 0.0200 | **0.7919 ± 0.0159** | **+0.0097 ± 0.0016** (t = 6.0) | 0.7908 ± 0.0236 | **0.8003 ± 0.0170** | **+0.0095 ± 0.0022** (t = 4.3) |
| MAGIC ℓ_p | 1e-6 | 5 | 0.7810 ± 0.0020 | 0.7814 ± 0.0020 | +0.0004 ± 0.0001 (t = 2.6) | 0.7811 ± 0.0026 | 0.7813 ± 0.0025 | +0.0002 ± 0.0002 (t = 1.2) |
| MAGIC ball | 1e-6 | 5 | 0.7810 ± 0.0020 | 0.7810 ± 0.0022 | +0.0001 ± 0.0001 (t = 0.4) | 0.7811 ± 0.0026 | 0.7810 ± 0.0027 | −0.0001 ± 0.0002 (t = −0.5) |

`U` (the true non-smooth potential) agrees: 400.9 → 396.0 and 390.8 → 386.5 on Titanic.

Three things changed for the better relative to the mini-batch runs: **`s = 5` now works
directly** on Titanic (no strength search needed), the wins are much stronger (t ≈ 6 rather than
3), and MAGIC's accuracy rises to 0.781 test from 0.774 — the intercept helps and removing
mini-batch noise removes a variance floor.

**The same caveat still applies.** At the selected `η` the Titanic chains are mid-transient
(reversible reaches 0.763 at η = 1e-5 versus 0.798 at η = 3e-5), which is where the metric has
headroom; MAGIC at η = 1e-6 is nearly converged (0.7810 versus 0.7829 at η = 1e-4), which is why
its gains are ~0.0004 at best. This remains a **convergence-rate** result, not evidence of a
better stationary law.

Also worth noting: at the selected `η` the projection rate is ≈ 0, so `K` is not binding and the
ball/ℓ_p difference enters **only** through `J`'s axes (`s·w` versus `−s·∇g`). That is why the
two MAGIC reversible arms are numerically identical while their non-reversible arms differ.

## Titanic at η = 3e-5 — the win disappears

`titanic_eta3e5.py`; table in `results/exact/titanic_eta3e-5.csv`, figures
`figures/exact/titanic_*_exact_anchored_eta3e-5.*`. `η` here is **specified, not selected**, so
there is no step-size selection bias. All strengths are reported (R = 150, seed 4100).

`η = 3e-5` is where Titanic has essentially converged — `η = 1e-4` gives the same place
(ball 0.806 test, `U` 383.4; ℓ_p 0.802 test, `U` 371.1).

| | s = 0 (control) | 0.25 | 1 | 2 | 5 |
|---|---|---|---|---|---|
| ball, paired Δ train | 0.00000 | −0.0003 (t = −0.9) | −0.0002 (t = −0.2) | +0.0001 (t = 0.1) | −0.0001 (t = −0.0) |
| ball, paired Δ test | 0.00000 | −0.0001 (t = −0.1) | +0.0005 (t = 0.6) | +0.0005 (t = 0.5) | +0.0015 (t = 1.3) |
| ℓ_p, paired Δ train | 0.00000 | +0.0004 (t = 1.0) | −0.0001 (t = −0.2) | −0.0006 (t = −1.0) | **−0.0016 (t = −2.4)** |
| ℓ_p, paired Δ test | 0.00000 | +0.0008 (t = 1.1) | +0.0003 (t = 0.3) | +0.0006 (t = 0.6) | +0.0004 (t = 0.4) |

The `s = 0` control returns exactly `0.00000`, which verifies the pairing. Every other cell is
inside the noise, except `s = 5` on the ℓ_p set, where the non-reversible arm is slightly
**worse** on training.

**The decisive comparison.** The *reversible* arm at `η = 3e-5` reaches **0.8048** (ball) and
**0.8043** (ℓ_p) test accuracy. The *non-reversible* arm at `η = 1e-5` — the cell that produced
the t ≈ 6 win — reaches only **0.7880** and **0.8003**. So the win at `η = 1e-5` was recovering
ground that a larger step size hands you for free, and it never catches up to simply running the
reversible chain at a well-chosen step.

This is the clearest available statement of the caveat attached to every earlier result in this
repo: the non-reversible drift buys **convergence rate within a fixed iteration budget at a
too-small step**, not a better answer. Where the chain converges inside the budget, the two arms
are indistinguishable, which is exactly what the theory says should happen — `J` is tangential
and conservative, and leaves the invariant law alone.

## The (η, s) landscape

`grid_study.py`; full table `results/grid/grid.csv`, per-cell curves `results/grid/curves_*.npz`,
20 figures in `figures/grid/` (same layout as before — training left, test right, common `[0,1]`
axis, zoom inset; one figure per experiment × η, with the reversible arm in blue and the
non-reversible arm at each `s` in a green ramp). Exact gradient, LASSO anchor, R = 100, seed 4100.
**The entire grid is reported — nothing was selected.**

Final test accuracy (rows = η, columns = s; "rev" = reversible):

| MAGIC ball | rev | 0.25 | 1 | 2 | 5 | | MAGIC ℓ_p | rev | 0.25 | 1 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| η=1e-6 | .7810 | .7809 | .7809 | .7809 | .7811 | | η=1e-6 | .7810 | .7811 | .7811 | .7814 | .7811 |
| η=3e-6 | .7835 | .7835 | .7835 | .7834 | .7833 | | η=3e-6 | .7839 | .7838 | .7836 | .7835 | .7786 |
| η=1e-5 | .7835 | .7836 | .7835 | .7834 | .7835 | | η=1e-5 | **.7841** | .7841 | .7840 | .7839 | **.5358** |
| η=3e-5 | .7834 | .7834 | .7835 | .7835 | **.6134** | | η=3e-5 | .7841 | .7841 | .7839 | **.5931** | **.5168** |
| η=1e-4 | **.7835** | .7836 | .7834 | .7601 | **.5660** | | η=1e-4 | .7841 | .7841 | **.6283** | **.5857** | **.5482** |

| Titanic ball | rev | 0.25 | 1 | 2 | 5 | | Titanic ℓ_p | rev | 0.25 | 1 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| η=1e-6 | .6036 | .6042 | .6050 | .6054 | .6084 | | η=1e-6 | .6341 | .6332 | .6320 | .6296 | .6325 |
| η=3e-6 | .7020 | .7017 | .7001 | .7023 | .7056 | | η=3e-6 | .7148 | .7149 | .7141 | .7176 | .7279 |
| η=1e-5 | .7702 | .7702 | .7722 | .7784 | .7867 | | η=1e-5 | .7921 | .7928 | .8003 | .7997 | .8003 |
| η=3e-5 | .8054 | .8050 | .8062 | .8065 | .8060 | | η=3e-5 | .8049 | .8059 | .8052 | .8051 | .8055 |
| η=1e-4 | **.8072** | .8070 | .8063 | .8053 | .8034 | | η=1e-4 | **.8053** | .8047 | .8036 | .8057 | .7808 |

Four things the landscape says:

1. **η governs accuracy; s does not.** Sweeping η moves Titanic test accuracy from 0.60 to 0.81.
   At any η where the chain converges, changing s moves it by ~0.001.
2. **s has a stability ceiling that depends on the geometry**, and past it accuracy collapses
   rather than degrades gracefully (0.78 → 0.52–0.63). The ceiling falls as η rises:
   MAGIC ℓ_p tolerates `s ≤ 5` at η = 1e-6 but only `s ≤ 0.25` at η = 1e-4. The ℓ_p set is far
   more fragile than the ball because `J`'s axis is `∇g` (‖∇g‖ ≈ 4.6 on ∂K) rather than `w`
   (‖w‖ ≈ 1.4), so the same `s` gives a ~3× stronger rotation.
3. **s only helps in a narrow band** — where the chain is still mid-transient at that η. Titanic
   at η = 1e-5 (`s = 5`: +0.0165 test on the ball, t = 6.2 on training); MAGIC at η = 1e-6
   (t = 2.1–4.2). Everywhere else it is neutral (small s) or destructive (large s).
4. **No cell beats the best reversible cell.** Best non-reversible vs best reversible, over the
   whole grid: MAGIC ball +0.0000, MAGIC ℓ_p −0.0000, Titanic ball −0.0001, Titanic ℓ_p +0.0006.
   The `s = 0` control returns exactly the reversible arm at every η, confirming the pairing.

So the usable recipe is: pick the largest η that is still stable, at which point s is free to be
small; the band where s pays is the band where η is too small.

## Per-block strengths `s = (s₁, s₂, s₃)`

`block_study.py`; table `results/blocks/blocks.csv`, triples `results/blocks/blocks_meta.json`,
16 figures in `figures/blocks/` (same layout). Exact gradient, LASSO anchor, R = 100, seed 4100,
all triples reported.

The three blocks are not interchangeable:

| | I₁ = w₁..w₃ | I₂ = w₄..w₆ | I₃ = w₇..w₉ |
|---|---|---|---|
| MAGIC | PC1–PC3, ‖∇‖ = 5762 | PC4–PC6, ‖∇‖ = 2292 | PC7–PC9, ‖∇‖ = 732 |
| Titanic | Age, SibSp, Parch, ‖∇‖ = 89 | Fare, is_female, Pclass_2, ‖∇‖ = 113 | Pclass_3, Emb_Q, Emb_S, ‖∇‖ = 131 |

Block `ℓ` contributes rotational drift `η·s_ℓ·r_ℓ·‖∇_{I_ℓ}U₀‖`, so a principled anisotropic rule
is **`s_ℓ ∝ 1/(r_ℓ‖∇_{I_ℓ}U₀‖)`** — every block contributes equally. That gives
`(1.04, 2.46, 11.5)` for MAGIC and `(8.35, 3.33, 3.33)` for Titanic at mean strength 5. The
**reversed** triple is run as a control: same total strength, wrong ordering.

### 1. The effect localises to one block

Titanic ball, η = 1e-5, paired Δ test (t):

| reversible | (5,5,5) | (5,0,0) | (0,5,0) | (0,0,5) |
|---|---|---|---|---|
| — | +0.0165 (4.9) | −0.0002 (−0.2) | **+0.0174 (6.2)** | +0.0051 (2.5) |

**Block 2 alone reproduces the entire isotropic effect**; block 1 contributes nothing. Block 2 is
Fare / is_female / Pclass_2 — the strongest predictors. The same holds on Titanic ℓ_p
((0,5,0): +0.0068, t = 3.3; (5,0,0) and (0,0,5) both null). This is why the hand-picked
`(2,7,2)` worked: it happened to boost block 2.

### 2. Instability also localises — to a different block

MAGIC ℓ_p, η = 1e-5, test accuracy (reversible 0.7841):

| (5,5,5) | (5,0,0) | (0,5,0) | (0,0,5) | balanced (1.04, 2.46, 11.5) |
|---|---|---|---|---|
| **0.5358** | **0.5483** | 0.7839 | 0.7841 | **0.7838** |

Isotropic `s = 5` collapses, and `(5,0,0)` shows **block 1 alone is the cause** — blocks 2 and 3
are perfectly stable at `s = 5`. The balanced triple carries `s₃ = 11.5`, more than double the
isotropic value that collapsed, and stays stable. **Anisotropic `s` buys back the stability
ceiling.**

### 3. The ordering matters, not just the magnitude

The reversed control has identical total strength and fails:

| | balanced | reversed | 
|---|---|---|
| MAGIC ℓ_p, η = 1e-6 | +0.0006 train (t = 5.0) | **−0.0801 (t = −7.9)** |
| MAGIC ball, η = 1e-5 | −0.0001 (t = −1.1) | **−0.2076 (t = −52)** |

So matching `s_ℓ` to each block's gradient scale is doing real work.

### 4. Best cells, and the unchanged caveat

Best paired test gain anywhere in the study: Titanic ℓ_p at η = 1e-5 with the balanced triple
`(8.32, 3.27, 3.42)`, **+0.0100 ± 0.0021 (t = 4.7)**, ahead of isotropic `(5,5,5)` at +0.0083 and
of `(2,7,2)` at +0.0093. On MAGIC ℓ_p at η = 1e-6 the balanced triple is also the best
(+0.0006 train, t = 5.0; +0.0005 test, t = 2.9 at mean strength 1).

At converged step sizes (Titanic η = 3e-5, MAGIC η = 1e-5) every triple is within noise of the
reversible arm, exactly as in the isotropic grid. Per-block tuning widens the usable strength
range and locates where the rotation does its work; it does not change the fact that the gain
lives in the transient.

## LASSO weight λ from 5 to 20 — does the non-reversible lead grow?

`lambda_lasso_study.py`; table `results/lambda/lambda_lasso.csv`, 40 figures in `figures/lambda/`
(same layout). Back to the `091b02b` setup: exact gradient, LASSO anchor, d = 10, R = 100,
seed 4100. λ ∈ {5, 7.5, 10, 15, 20} absolute (at `091b02b` the rule `λ = 0.01 n_train` gave 152.2
on MAGIC and 7.12 on Titanic, so this range puts both datasets on the same footing). Two step
sizes per dataset — transient and converged — and `s ∈ {1, 5}`. Everything reported.

**Coupling to declare:** `δ = log2/(9λ)` is kept, so `a ∈ [½,1]` at every λ. Raising λ therefore
also sharpens the smoothing (δ falls 0.0154 → 0.00385). The two cannot both be fixed while
holding the anchor bound constant; δ is in every row of the CSV.

### Yes — on Titanic, at the transient step, the lead grows monotonically with λ

Paired Δ test accuracy (t), η = 1e-5:

| | λ = 5 | 7.5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Titanic ball, s = 5 | +0.0162 (4.8) | +0.0163 (4.8) | +0.0175 (5.1) | +0.0203 (5.9) | **+0.0240 (6.4)** |
| Titanic ball, s = 1 | +0.0016 (0.6) | +0.0018 (0.7) | +0.0022 (0.8) | +0.0063 (2.4) | +0.0075 (2.8) |
| Titanic ℓ_p, s = 1 | +0.0088 (4.3) | +0.0089 (4.4) | +0.0098 (4.8) | +0.0122 (5.6) | **+0.0142 (5.9)** |
| Titanic ℓ_p, s = 5 | +0.0082 (3.6) | +0.0084 (3.8) | +0.0092 (3.9) | +0.0097 (3.9) | +0.0152 (5.4) |

The lead roughly **doubles** from λ = 5 to λ = 20. On MAGIC it is flat (ball s = 1 holds at
+0.0003, t ≈ 2.0 across the whole range; ℓ_p s = 5 at +0.0005–0.0006, t ≈ 2).

The mechanism is visible in the "near-kink" column: the number of coefficients with |w_j| < 0.05
rises from ~1.2 to ~2.8 on Titanic as λ grows. More coordinates sit at kinks, which is exactly
the regime the anchor is built for, and the rotation helps the chain keep moving there.

### But the lead grows because the reversible arm degrades faster, not because the non-reversible arm improves

Absolute test accuracy, Titanic ℓ_p, η = 1e-5:

| | λ = 5 | 7.5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| reversible | **0.7920** | 0.7915 | 0.7900 | 0.7848 | 0.7778 |
| non-rev, s = 1 | **0.8008** | 0.8004 | 0.7998 | 0.7970 | 0.7920 |

Both arms fall with λ; the reversible one falls faster. The non-reversible arm at λ = 20 (0.7920)
only matches the reversible arm at λ = 5 (0.7920). So the widening gap is not worth anything in
absolute terms — the best accuracy is still at the **smallest** λ.

Best absolute cell anywhere in the study, non-reversible vs best reversible: MAGIC ball +0.0001,
MAGIC ℓ_p +0.0001, Titanic ball −0.0000, Titanic ℓ_p +0.0011. And at the converged step
(η = 3e-5) there is no lead at any λ — every |t| < 2.8 and most < 1.5.

**Answer:** raising λ does give the non-reversible arm a larger and cleaner lead (t up to 6.4),
and it is the one knob so far that widens the gap systematically rather than by luck. It buys
that lead by making the problem harder for both arms, so it does not produce a better model.

---

# Can the non-reversible arm beat the reversible one? A 7-hypothesis investigation

Scripts `probe_*.py`, results in `results/probe_*/`. Seven hypotheses were probed in parallel and
each claimed win was handed to an adversarial agent told to refute it. Six probes completed
(`calibration` and three refutations hit a session limit and are marked UNREFUTED below).

## The constraint that shapes the answer

`J` is skew, `div J = 0`, and `Jn = 0` on `∂K`. So **both arms have the same invariant law.** No
tuning can separate them on any *stationary* quantity — last-iterate accuracy included. That is a
theorem, not an empirical finding, and it is why ten rounds of tuning η, s, Λ, per-block s and λ
never produced an accuracy win that survived best-vs-best.

But two theorems guarantee a difference elsewhere: the asymptotic variance of ergodic averages is
never larger for the non-reversible chain (Hwang–Hwang–Sheu; Duncan–Lelièvre–Pavliotis), and the
spectral gap is never smaller. **The win has to be measured on a time-average, not a marginal.**

## Results

| hypothesis | verdict | best-vs-best | refutation |
|---|---|---|---|
| **mixing** (ESS per second) | **WIN** | **yes** | **survives** |
| hessian (MSE of ergodic average) | PARTIAL, win on the metric | yes | unrefuted |
| multimodal (harder target, fixed budget) | WIN at fixed budget | yes | unrefuted |
| ergodic (ergodic-average accuracy) | PARTIAL | no | unrefuted |
| overdispersed (hard initialisation) | not a win | no | refuted |
| wallclock (compute-matched accuracy) | BACKFIRED | no | — |
| calibration | did not run | — | — |

### The defensible win: sampling efficiency

Titanic, η = 1e-4, s = 5, R = 96 **coupled** chains, wall-clock corrected (the non-reversible arm
is charged its full 1.13× per-step cost). IACT falls on all nine coordinates; ESS per second:

| test fn | IACT rev → nrev | ESS/s rev → nrev | speed-up |
|---|---|---|---|
| w4 | 111 → 17 | 3.26 → 17.8 | **×5.47** |
| w6 | 107 → 16 | 3.41 → 18.0 | ×5.30 |
| w9 | 174 → 48 | 2.16 → 6.54 | ×3.03 |
| w8 | 111 → 33 | 3.29 → 9.00 | ×2.74 |
| w7 (pre-registered primary) | 190 → 130 | 1.99 → 2.62 | ×1.32 |
| ‖w‖² | 12.3 → 12.0 | 29.2 → 26.4 | **×0.91 (worse)** |

Summed over 12 training-only test functions: ×1.56. Replicated on the smoothed ℓ_p geometry
(×1.64) and on three further seeds.

**What the adversarial re-check established.** (i) The `s = 0` pairing control is **bit-identical**
between the arms (max diff exactly 0), so the paired SEs are real. (ii) An independent 4th seed
reproduces every coordinate and every backfire. (iii) With both arms started from a common
*pre-equilibrated* ensemble and no burn-in discarded, the win persists (×1.21, t = 6.9) — it is not
a transient. (iv) **Best-vs-best survives at matched discretisation bias**: letting each arm pick
its own (η, s) under a bias ceiling measured against an η→0 reference, the non-reversible arm wins
at every ceiling tried (×1.21, ×1.36, ×1.14).

**What the re-check knocked down.** The headline ×1.31 is at the optimistic end of its own sampling
distribution. Under a truncation-free estimator (ESS from the across-chain variance of chain
time-averages, immune to Geyer truncation) the pre-registered primary function improves only
**×1.12, bootstrap 95% CI [0.97, 1.30]** — not individually significant. Pooling six independent
measurements gives **≈ ×1.2**, not ×1.31. The large per-coordinate gains (×2.7–×5.5) are not in
dispute; the modest gain on the *worst* coordinate is what sets the honest headline.

### Supporting evidence, same direction

* **Monte-Carlo variance of ergodic averages** falls by 2.0–2.6× (test-set predicted probability
  variance ratio rev/nrev = 2.59 on the ball, 2.43 on ℓ_p), and survives a common-start control
  that removes initial-condition spread (ratio 2.32).
* **MSE of the ergodic average at best-vs-best**, each arm at its own training-selected η,
  confirmation seed, R = 400: non-reversible 8.03e-6 vs reversible 1.157e-5 on the test predicted
  probability, paired t = −3.54. (Unrefuted.)

### Where it does not win, stated plainly

* **Prediction.** Last-iterate test accuracy: t = +1.37 on the ball, t = −0.02 on ℓ_p — nothing, as
  the shared-invariant-law argument requires.
* **Best-vs-best predictive quality.** The reversible arm at its own best η = 1e-3 reaches test
  accuracy 0.8259 / log loss 0.4884, beating the non-reversible arm's 0.8180 / 0.4900 at η = 3e-5.
  (Caveat in both directions: at η = 1e-3 the reversible chain is clipped to the boundary on 73% of
  steps — it scores well predictively while being a badly biased sampler of `π_K`.)
* **Compute-matched accuracy** depends on an implementation artefact: the cost ratio is 2.77 at
  R = 1 chain but 1.01 at R = 600, against a break-even of 1.15–1.28. Vectorised, it wins; single
  chain, it loses.
* **Not every test function improves.** ‖w‖² is consistently ×0.91, and the likelihood functionals
  are neutral or slightly worse.

## Bottom line

**Yes — the non-reversible arm beats the reversible one, decisively and reproducibly, on the thing
non-reversibility is for: Monte-Carlo efficiency per unit wall clock (≈ ×1.2 on the worst
coordinate, ×2.7–5.5 on the best, ×1.56 summed) and asymptotic variance of posterior functionals
(2.0–2.6× lower).** It does **not** beat it on predictive accuracy, and it cannot: the two chains
sample the same law. Anyone wanting posterior functionals — credible intervals, predictive
variances, expectations — should use the non-reversible arm. Anyone wanting a point prediction
gains nothing from it.

## Figures for the mixing win

`mixing_figures.py` → `figures/mixing/` (300-dpi PNG + vector PDF each). All but the
autocorrelation panel are rebuilt from the saved probe outputs; the ACF panel re-runs two short
coupled chains. Palette: reversible `#1f5fbf` (circle), non-reversible `#1a9850` (square) —
two categorical slots, validated (OKLab ΔE 27.0 normal / 25.6 deutan, both inside the lightness
band, both ≥ 3:1 on the surface); marker shape is a secondary encoding so identity never rests
on colour alone.

| figure | what it shows |
|---|---|
| `mixing_ess_iact` | The headline. Per test function: IACT (lower better) and ESS per second (higher better), as paired dumbbells, with the speed-up direct-labelled. Eleven of twelve functions improve; `‖w‖²` is the one that does not. |
| `mixing_acf` | The mechanism. Autocorrelation functions for the largest gain (`w₄`), the worst case (`w₇`) and the backfire (`‖w‖²`). `w₄` *oscillates* under the non-reversible dynamics — the rotation makes the chain circulate rather than diffuse — and crosses zero within ~50 iterations instead of decaying over ~300. |
| `mixing_forest_w7` | The honesty panel. Eight measurements of the same quantity for the worst-case coordinate: six Geyer estimates across four seeds and two burn-ins, their pooled value (×1.23), and a truncation-free estimator whose bootstrap 95% interval **includes 1**. |
| `mixing_bias_frontier` | The fairness panel. ESS/s against discretisation bias, both arms free to pick their own η. Read vertically at any bias level the green curve sits above the blue one — comparing points at *different* bias is not a fair comparison. |
| `mixing_rmse_budget` | The caveat. RMSE of the posterior mean against compute budget, both arms re-optimised at each budget. Non-reversible is ×1.33 ahead at short budgets; the curves cross near T = 10 and the reversible arm wins beyond it. |
| `mixing_accuracy_ball` | Accuracy at *exactly* the mixing-win configuration. The two curves sit on top of each other: the gap is **0.08 percentage points** of test accuracy (final iterate −0.0038, t = −2.6), in the non-reversible arm's slight disfavour. This is what a shared invariant law looks like. |
| `mixing_accuracy_lp` | The same plot on the smoothed ℓ_p set — and here the same `s = 5` costs **2.3 percentage points** of test accuracy (−0.0245, t = −7.0; training −0.0355, t = −12.4). On this geometry `J`'s axis is `∇g` rather than `w`, so the same strength gives a ~3× stronger rotation and a correspondingly larger discretisation bias. **The ℓ_p mixing win is real but it is not free.** |
