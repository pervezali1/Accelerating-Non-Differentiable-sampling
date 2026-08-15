# Accelerating-Non-Differentiable-sampling

## Anchored Langevin on a univariate Laplace target

`anchored_langevin_laplace_1d.ipynb` runs the protocol of `beta_nonrever_Gaunssian.ipynb`
(N-particle ensemble from a common start, W1 to reference samples recorded every iteration,
density plots of the terminal ensemble) with **anchored Langevin** on the univariate Laplace
target `pi(x) ∝ exp(-|x-mu|/b)`, which is non-differentiable at `x = mu`.

Anchored dynamics, for a smooth anchor `U0` and `Delta = U - U0`:

```
dX_t = -grad U0(X_t) exp(Delta(X_t)) dt + sqrt(2) exp(Delta(X_t)/2) dW_t
```

`pi ∝ exp(-U)` is invariant for any anchor (the notebook derives it from `b p = grad(D p)`),
and `U` is used only through the scalar weight `exp(Delta)` — `grad U` is never evaluated.

Anchors compared, against a subgradient-Langevin baseline: `U0 = 0`, quadratic
`U0 = (x-mu)^2/(2 tau^2)`, and pseudo-Huber `U0 = sqrt((x-mu)^2 + delta^2)/b`.

Experiments:

1. **Laplace on R** — W1 curves and terminal densities for `b ∈ {0.5, 1, 2}` (the univariate
   counterpart of the notebook's dim1/dim2/dim3 panels), plus an explicit demonstration that the
   zero anchor diverges when the state space is unbounded.
2. **Truncated Laplace on `[-R, R]`** — the projected/constrained setting of the Gaussian
   notebook, with reference samples by rejection; all four methods, including the zero anchor,
   which is well behaved on a compact set.
3. **Discretization bias vs stepsize** — W1 at stationarity over `eta ∈ [0.005, 0.2]`, 3 runs each.

Headline results: all anchors reproduce the target without `grad U`; the quadratic anchor is the
fastest to converge on the slowest-mixing target (`b = 2`: W1 0.147 vs 0.265 for the subgradient
baseline at iteration 2000); the pseudo-Huber anchor keeps `Delta` bounded, needs no clamping, and
is the most accurate method in the truncated experiment and at the largest stepsizes.

## Non-reversibility: constant skew J vs J = 0

`anchored_langevin_nonreversible.ipynb` adds a constant skew-symmetric `J` (the Gaussian notebook's
`J_fun`) to the anchored dynamics:

```
dX = -(I + J) grad U0(X) exp(Delta(X)) dt + sqrt(2) exp(Delta(X)/2) dW
```

The notebook proves invariance is unchanged for any `a` (the extra flux is `J grad(exp(-U0))`, and
`div(J grad phi) = 0` for constant skew `J`), so `J` alters the rate only.

Two facts shape the experiments. First, **in d = 1 every skew-symmetric matrix is zero**, so a
constant-J scheme applied to a univariate target *is* the J = 0 scheme — the state space has to grow.
Second, the non-reversible drift is `-J grad U0 exp(Delta)`, so it acts through the **anchor's**
gradient: with `U0 = 0` the value of `a` is irrelevant.

1. **Product Laplace in d = 3**, `b = (0.5, 1, 2)`: `{subgradient LMC, anchored} x {J = 0, J_a}`, plus a
   sweep over `a` and an `eta`-refinement check.
2. **Univariate Laplace via a 2D lift**: sample `pi(x1) pi(x2)` with skew coupling, read the x1 marginal.

Results: `J` accelerates the slow coordinate (d = 3, coordinate 3: W1 0.212 -> 0.133 at `a = 2`), leaves
the stationary law untouched, and costs discretization accuracy on the fast coordinates at large `a` —
recoverable by shrinking `eta`. The lift gives the biggest win on a genuinely univariate target:
W1 0.272 -> 0.071 at `a = 2`, for one extra scalar per particle.

Figures are written to `figures/`. Requires `numpy`, `scipy`, `torch`, `matplotlib`, `seaborn`.
