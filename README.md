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
