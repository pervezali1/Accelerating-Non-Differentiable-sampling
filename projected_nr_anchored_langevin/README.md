# Projected Non-Reversible Anchored Langevin (PNRAL) on the 3-D ball

Sampling `pi_C(x) ∝ exp(-U(x)) 1{||x|| <= R}` on `C = {x ∈ R^3 : ||x||_2 <= R}`
for a **non-differentiable** `U`, by anchoring `U` on a smooth surrogate `U0`
and adding a state-dependent skew-symmetric (non-reversible) drift.

## Method

| ingredient | expression | purpose |
|---|---|---|
| anchor coefficient | `a(x) = exp(U(x) − U0(x))` | carries all non-smoothness in a *scalar*; the only gradient ever evaluated is `∇U0` |
| drift | `b(x) = −a(x) (I + α J_s(x)) ∇U0(x)` | reversible descent + non-reversible rotation |
| skew matrix | `J_s(x) = s·[x]_×` | `J^T = −J`, `div J = 0`, `x^T J = 0` |
| update | `y = x + h b + sqrt(2 h a) ξ`, `x' = Proj_C(y)` | Euler–Maruyama + metric projection |

Why no correction terms are needed:

* **No `∇U`.** With `p ∝ exp(−U)` we get `a p ∝ exp(−U0)`, so
  `a ∇U0 p + ∇(a p) = ∇U0 e^{−U0} − ∇U0 e^{−U0} = 0`: the reversible flux
  vanishes exactly, for any smoothing level. No `div(a)` drift correction.
* **No `div(J)`.** The extra flux is `α J ∇(e^{−U0})`, whose divergence is
  `α (div J)·∇q + α tr(J ∇²q) = 0` — the first term by `div J = 0`, the second
  because `tr(skew · symmetric) = 0`.
* **Boundary compatible.** `J_s(x) g = s (x × g) ⊥ x`, so the non-reversible
  drift is tangent to every sphere `||x|| = c`, in particular to `∂C`: it
  transports mass *around* the constraint instead of into it.

Only the product `α·s` sets the perturbation strength, so `s = 1` is fixed and
`α` is tuned.

## Test problem

`U(x) = |x1|+|x2|+|x3|` (Laplace, truncated to the ball), anchored with
`U0(x) = Σ_j sqrt(x_j² + δ²)`, giving `log a(x) = Σ_j (|x_j| − sqrt(x_j²+δ²)) ∈ [−3δ, 0]`,
hence `exp(−3δ) ≤ a(x) ≤ 1`. `log a` is always formed in the log domain — never
as `exp(U)/exp(U0)`.

## Run

```bash
pip install numpy scipy pandas matplotlib seaborn arviz
python pnr_anchored_langevin_ball3d.py            # full experiment (~4 min)
python pnr_anchored_langevin_ball3d.py --quick    # smoke run (~20 s)
```

Writes figures + diagnostic CSVs to `pnral_outputs/`. `run_sampler(...)` returns
`(samples, per-iteration DataFrame, meta)`.

## Results (R=3, δ=0.05, h=0.02, s=1, 4 chains × 40 000 iterations)

Structural identities hold to machine precision (`max|J+Jᵀ| = 0`,
`max|xᵀJ| = 0`, `max|div J| = 0`, tangency `1.4e-14`); `a ∈ [0.8607, 0.9978]`
against the bound `[exp(−3δ), 1] = [0.86071, 1]`; `max ||x_k|| = R` exactly in
every chain; no NaN/inf.

| α | ESS (min) | ESS/s | IACT | R-hat | projected | KS(x1) vs exact |
|---|---|---|---|---|---|---|
| 0   |  998 | 125 | 62.0 | 1.0044 |  3.0% | 0.030 |
| 0.5 | 1029 | 130 | 60.6 | 1.0041 |  3.0% | 0.030 |
| 1   | 1093 | 147 | 56.6 | 1.0030 |  3.0% | 0.030 |
| 2   | 1303 | 163 | 48.1 | 1.0035 |  3.1% | 0.029 |
| 4   | 1657 | 212 | 36.5 | 1.0021 |  4.2% | 0.029 |
| 8   | 2614 | 385 | 23.4 | 1.0026 | 10.5% | 0.070 |

The non-reversible drift gives a monotone ESS gain (**3.1× ESS/s at α = 8**),
but large α costs accuracy at fixed `h`: Euler integrates the tangential
rotation along chords rather than arcs, inflating `||x||`, so the projection
rate jumps to 10.5% and the boundary is over-populated (visible as the red
shell in `samples_3d_sphere.png`). α ≈ 2–4 buys ~1.3–1.7× ESS at unchanged bias.

Step-size study at **equal simulated time** `n_iter·h = 800` (halving `h`
doubles `n_iter`, otherwise a smaller `h` merely explores less and its accuracy
gain is masked by Monte-Carlo error):

| α | h | n_iter | ESS | ESS/s | projected | KS(x1) |
|---|---|---|---|---|---|---|
| 0 | 0.02  |  40 000 | 1032 | 145 | 2.97% | 0.0305 |
| 0 | 0.01  |  80 000 | 1192 |  84 | 2.12% | 0.0256 |
| 0 | 0.005 | 160 000 | 1076 |  33 | 1.59% | 0.0124 |
| 2 | 0.02  |  40 000 | 1331 | 166 | 3.14% | 0.0295 |
| 2 | 0.01  |  80 000 | 1420 |  89 | 2.17% | 0.0207 |
| 2 | 0.005 | 160 000 | 1309 |  45 | 1.59% | 0.0059 |

ESS at fixed simulated time is essentially flat while the discrepancy against
an exact rejection sample shrinks with `h` — i.e. the residual error is the
`O(h)` discretisation/projection bias, not the anchoring, which is exact for
any `δ`.
