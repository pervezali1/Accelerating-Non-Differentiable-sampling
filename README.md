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

Figures are written to `figures/`. Requires `numpy`, `scipy`, `torch`, `matplotlib`, `seaborn`.
