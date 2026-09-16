# Accelerating Non-Differentiable Sampling

Numerical experiments for **non-reversible anchored Langevin dynamics (NALD)** on targets whose
potential is only locally Lipschitz.

## The dynamics

For a target $\pi \propto e^{-U}$ with $U$ locally Lipschitz, a $C^2$ *anchor* $U_0$ and a $C^2$
field $\psi$, set

```
a(x)       = exp(U(x) - U0(x))
c(x)       = exp(U(x)) J(x) grad psi(x)
b_alpha(x) = -a(x) grad U0(x) + alpha c(x)

dX_t = b_alpha(X_t) dt + sqrt(2 a(X_t)) dW_t                                   (NALD)
```

With the canonical choice `psi = exp(-U0)` this becomes

```
dX_t = -a(X_t) (I + alpha J(X_t)) grad U0(X_t) dt + sqrt(2 a(X_t)) dW_t        (NALD-c)
```

`grad U` is never evaluated — only the smooth anchor's gradient is. The reversible part has
*identically zero* probability flux against `exp(-U)`, and the antisymmetric part preserves
invariance whenever `J` is skew-symmetric with divergence-free columns.

## Notebook

[`notebooks/nald_laplace_experiments.ipynb`](notebooks/nald_laplace_experiments.ipynb) applies NALD in
$d = 3$ to two non-differentiable targets and compares three perturbations.

**Targets** (each defined in its own cell, with $U$ and $U_0$ written out explicitly):

| | $U(x)$ | anchor $U_0(x)$ | non-smooth set |
|---|---|---|---|
| A — elliptical Laplace | $\sqrt{x^\top\Sigma^{-1}x}$ | $\sqrt{x^\top\Sigma^{-1}x+\delta^2}$ | $\{0\}$ (conical cusp) |
| B — $\ell^1$ Laplace | $\sum_i \lvert x_i\rvert/b_i$ | $\sum_i \sqrt{x_i^2+\delta^2}/b_i$ | $\bigcup_i\{x_i=0\}$ |

Both anchors are hyperbolic (pseudo-Huber) smoothings, so `a = exp(U - U0)` is bounded above and below
and `grad U0` is bounded and Lipschitz. Both targets admit exact i.i.d. sampling, which is used as
ground truth.

**Perturbations**: `J = 0` (reversible), a constant skew-symmetric `J_a = hat(u)` with
`u = (1,1,1)/sqrt(3)`, and the state-dependent

```
          [   0    -s x3   s x2 ]
J_s(x) =  [  s x3    0    -s x1 ]  =  s * hat(x),     J_s(x) v = s (x cross v)
          [ -s x2   s x1    0   ]
```

**What the notebook reports**: verification of the skew / divergence-free conditions and of
`div(J grad psi) = 0`; correctness against exact draws (marginals, Q-Q, moments) and independence of
the invariant law from the smoothing parameter `delta`; a paired stationarity-drift measurement of the
discretisation bias as a function of `alpha` and `h`; integrated autocorrelation times, ESS, speed-ups
and ergodic-average MSE at matched cost; and an analysis of *why* the two `J`s behave differently.

Speed-ups are never reported on their own — the measured discretisation bias is printed next to every
one of them, and configurations whose bias exceeds the tolerance are flagged, because a large `alpha`
at fixed `h` buys apparent mixing with real bias.

Findings worth flagging:

* Plain Euler–Maruyama is **unstable** for `J_s`. The exact `J_s` flow is a rotation
  `dx/dt = omega x x` with `omega = alpha s a grad U0` and conserves `|x|`, but an explicit Euler step
  inflates `|x|` by `sqrt(1 + (h |omega|)^2)` against a *bounded* restoring drift, so the chain escapes.
  The notebook uses a Lie–Trotter split with the rotation integrated exactly (Rodrigues), at the same
  one-gradient-per-step cost.
* The `J_s` perturbation `alpha c = (alpha s a grad U0) x x` **vanishes wherever `x` is parallel to
  `grad U0(x)`** — for target A that is exactly the principal axes of `Sigma`, i.e. the slow directions.
  This is why the constant `J_a` is the better accelerator of the slow coordinate here, while `J_s`
  mostly stirs the fast ones.
* `delta` cannot bias the answer, so it is free to tune — but it is not a free lunch. Small `delta`
  keeps `a = exp(U - U0)` near 1 and mixes fast while stiffening `grad U0`; large `delta` conditions the
  drift but collapses `a` and throttles the diffusion.

## Running

```bash
pip install numpy scipy matplotlib jupyter
jupyter lab notebooks/nald_laplace_experiments.ipynb
```

The notebook is self-contained (NumPy + Matplotlib only) and takes roughly 35 minutes to execute end
to end on a single core; it is committed with outputs, so it can be read without running anything.
