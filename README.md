# Accelerating Non-Differentiable Sampling

Anchored Langevin sampling for potentials that are only ever **evaluated**, never
differentiated, with an antisymmetric (skew) drift field added for circulation — and a hard
convex constraint handled by projection.

This repository carries the method from the synthetic 3-dimensional study in
`Constrained_Sampling_ANDS_ball_constrained.ipynb` over to two real binary-classification
posteriors:

| dataset | source | n | d | constraint radius R |
|---|---|---|---|---|
| **Titanic** | Kaggle training split (891 passengers) | 891 | 10 | 1.4 |
| **MAGIC Gamma Telescope** | UCI `magic04` (stratified 2 000-row subsample of 19 020) | 2 000 | 11 | 1.9 |

The main artefact is [`notebooks/Constrained_Sampling_ANDS_real_data.ipynb`](notebooks/Constrained_Sampling_ANDS_real_data.ipynb).

## The problem

For labels $y_i \in \{-1,+1\}$ and standardised features $\phi_i$, write $\psi_i = y_i \phi_i$.
The Gibbs posterior over classifier weights is

$$U(w) = \tau\,\frac{1}{n}\sum_i \max(0,\,1 - w^\top\psi_i) \;+\; \lambda\lVert w\rVert_1,
\qquad \pi(w) \propto e^{-U(w)}\mathbf 1_K(w),\qquad K = \{\lVert w\rVert_2 \le R\}.$$

Both terms are non-differentiable — the hinge kinks at every active margin, the $\ell_1$ prior on
every coordinate axis — so the sampler works from a smoothed *anchor* $U_0 \ge U$ and the time
change $e^{\Delta}$, $\Delta = U - U_0$. The hinge term is an **average**, not a sum, which keeps
the anchor gap $O(\tau\delta)$ instead of $O(n\tau\delta)$; with a sum, $e^{\Delta}$ underflows on
any real dataset.

## What the real-data setting adds

The dynamics for a state-dependent skew field is

$$dX_t = e^{\Delta(X_t)}\Bigl[-\bigl(I + J(X_t)\bigr)\nabla U_0(X_t) + \nabla\!\cdot J(X_t)\Bigr]dt
+ \sqrt{2}\,e^{\Delta(X_t)/2}\,dW_t,$$

and projection onto $K$ imposes a second condition, $J(x)\nu(x) = 0$ on $\partial K$.

The 3-dimensional notebook used the axial field $J_s(x)w = s\,(x \times w)$, which is a cross
product — it exists only in $d = 3$, and it is divergence-free, so the correction term
$\nabla\!\cdot J$ was derived but never needed. Its $d$-dimensional replacement, for a fixed
antisymmetric $A$,

$$J_s(x) = \frac{s}{R^2}\Bigl[\lVert x\rVert^2 A + x\,(Ax)^\top - (Ax)\,x^\top\Bigr],
\qquad \nabla\!\cdot J_s(x) = -\frac{s}{R^2}(d-2)\,Ax,$$

still annihilates $x$ identically (so $J_s\nu = 0$ on the wall by construction) but is **not**
divergence-free once $d > 2$. At $d = 10$ and $d = 11$ the correction is the same order as the
drift, so the notebook can run the ablation the synthetic study could not.

## What the runs show

Scored against two independent exact random-walk Metropolis references (acceptance 0.240 and
0.234), with the $W_1$ floor — the level at which two samples of this size are
indistinguishable — measured rather than assumed.

**Head-to-head at skew strength $s = 4$.** The constant field is antisymmetric and
divergence-free; it fails only $J\nu = 0$, and that one defect costs a factor of six.

| | mean $W_1$ | max KS | mass near $\partial K$ |
|---|---|---|---|
| Titanic — $J = 0$ | 0.0061 | 0.037 | 6.80% |
| Titanic — constant $J_a$ | **0.0387** | **0.218** | **16.50%** |
| Titanic — state-dependent $J_s$ | 0.0065 | 0.038 | 7.20% |
| Titanic — reference (truth) | 0.0063 | — | 0.56% |
| MAGIC — $J = 0$ | 0.0096 | 0.041 | 3.75% |
| MAGIC — constant $J_a$ | **0.0409** | **0.193** | **10.30%** |
| MAGIC — state-dependent $J_s$ | 0.0092 | 0.045 | 3.50% |
| MAGIC — reference (truth) | 0.0084 | — | 0.32% |

**Dropping $\nabla\!\cdot J$ is worse than using an inadmissible field** — the experiment the
3-dimensional study could not run, since there the correction was identically zero.

| mean $W_1$ | $s=0$ | $s=1$ | $s=2$ | $s=4$ | $s=8$ |
|---|---|---|---|---|---|
| Titanic — $J_s$ with the correction | 0.0061 | 0.0059 | 0.0055 | 0.0065 | 0.0131 |
| Titanic — $J_s$, correction dropped | 0.0061 | 0.0284 | 0.0490 | 0.0724 | **0.1014** |
| MAGIC — $J_s$ with the correction | 0.0096 | 0.0097 | 0.0090 | 0.0092 | 0.0140 |
| MAGIC — $J_s$, correction dropped | 0.0096 | 0.0353 | 0.0623 | 0.1047 | **0.1534** |

The two failures leave different fingerprints. The uncorrected field *is* tangential, so its
near-boundary mass barely moves (6.80% → 10.20% across the sweep on Titanic, against
6.80% → 27.15% for the constant field); its damage is interior, where the flux it generates is
no longer divergence-free. Boundary mass diagnoses a violated wall condition and is blind to a
missing correction term.

**Accuracy sees none of it.** Across the whole sweep — mean $W_1$ ranging over a factor of 16 —
every method's classification accuracy stays inside the reference posterior's own 5-95% spread.
On MAGIC the per-draw accuracy actually *rises* with the bias (0.7832 → 0.7855 for the constant
field, 0.7832 → 0.7859 for the uncorrected one, as $s$ goes 0 → 8) while the admissible field
sits on the reference at 0.7831: the two most badly biased samplers score highest.
$\operatorname{sign}(w^\top\psi_i)$ is invariant under $w \mapsto cw$, so a scale-invariant
statistic is structurally blind to the radial distortion these fields produce.

| | catches a violated $J\nu = 0$ | catches a missing $\nabla\!\cdot J$ |
|---|---|---|
| mass near $\partial K$ | yes | **no** |
| accuracy | **no** | **no** |
| $W_1$ against a trusted reference | yes | yes |

These accuracy numbers are **in-sample** — the posterior is fitted to all $n$ rows, and the
question asked is whether sampling bias reaches a decision statistic, not how well the model
generalises.

**Refining the stepsize fixes the projection atom and cannot fix a wrong boundary condition.**
At fixed physical time $T = 0.6$ on Titanic, the $J = 0$ atom falls 6.80% → 4.50% → 3.10% as
$\eta$ is refined 4×, with atom$/\sqrt\eta$ nearly constant (3.93, 3.67, 3.58), while the
constant-$J$ $W_1$ plateaus at 0.0387 → 0.0371 → 0.0397.

## Layout

```
ands/
  data.py          Titanic and MAGIC loaders (cached under data/)
  targets.py       the hinge + L1 Gibbs posterior, its anchor and projection
  skew.py          the three fields, closed-form divergences, autograd checks
  samplers.py      projected anchored Langevin; exact random-walk Metropolis reference
  diagnostics.py   W1 per coordinate, KS, boundary mass, reference floor
  experiments.py   the experiment grid and its result cache
  plots.py         figures
notebooks/         the executed study
tests/             what the theory claims, asserted
```

## Running it

```bash
pip install numpy scipy pandas matplotlib seaborn torch jupyter pytest
pytest tests/ -q
jupyter nbconvert --execute --inplace notebooks/Constrained_Sampling_ANDS_real_data.ipynb
```

Results are cached in `results/` keyed by a hash of their settings, so re-execution is cheap and
changing a setting invalidates only what it touches. Raw data is cached in `data/`; the loaders
fetch it once from public mirrors.
