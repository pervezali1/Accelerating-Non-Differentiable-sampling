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

The non-reversible drift gives a monotone ESS gain (**2.7x ESS/s at α = 8**),
but large α costs accuracy: Euler integrates the tangential rotation along
chords rather than arcs, inflating `‖x‖`, so the projection rate reaches 10.5%
and the boundary is over-populated (the red shell in `samples_3d_sphere.png`).

## Wasserstein accuracy

KS is the sup-gap between CDFs and saturates; W measures how far mass actually
moves, which is precisely this algorithm's failure mode. All configurations are
scored on the same `n_w = 20 000` draws against the same frozen reference
subsample (empirical W shrinks with n, so unequal n would be unfair), and the
floor sample is drawn from a pool disjoint from that subsample.

Separating bias from Monte-Carlo error: each statistic is recomputed **per
chain**, and the ratio `W(pooled) / mean_c W(chain c)` is compared against the
same ratio measured on exact i.i.d. samples (**0.587** here — not 1/2, because
the reference side keeps size `n_w` and `E[W₁] ~ sqrt(1/n + 1/m)` puts it near
`sqrt(2/5)`). Ratio near 0.59 ⇒ the number is noise; near 1.0 ⇒ real bias. The
chains must be independent replicates for this: disjoint quarters of one pooled
subsample share the parent chain's deviation from π, which does not cancel and
pins the ratio at 1 regardless of the truth.

| α | ESS/s | KS(x1) | W₁(radius) ± SE | ratio | W₁(coord) | ratio | exact W₂ |
|---|---|---|---|---|---|---|---|
| 0   | 125 | 0.0305 | 0.049 ± 0.018 | 0.95 | 0.047 | 0.74 | 0.272 |
| 0.5 | 135 | 0.0297 | 0.053 ± 0.020 | 0.99 | 0.048 | 0.74 | 0.269 |
| 1   | 155 | 0.0305 | 0.056 ± 0.019 | 0.99 | 0.049 | 0.79 | 0.265 |
| 2   | 158 | 0.0295 | 0.056 ± 0.011 | 1.00 | 0.042 | 0.73 | 0.267 |
| 4   | 220 | 0.0286 | 0.141 ± 0.022 | 1.00 | 0.075 | 0.91 | 0.306 |
| 8   | 319 | 0.0699 | 0.490 ± 0.027 | 1.00 | 0.233 | 0.99 | 0.589 |

i.i.d. floors at `n_w`: W₁(coord) 0.0117, W₁(radius) 0.0073, SW₁ 0.0116,
exact W₂ 0.262. SE is the across-chain spread of the pooled estimate.

What this shows:

* **The bias is radial, not marginal.** W₁ on the radius has ratio 0.95–1.00 at
  every α, so it is genuine bias even at α = 0 — the projection itself biases
  the radial distribution. The coordinate marginals sit at 0.73–0.79, much
  closer to the 0.587 noise floor, i.e. largely unbiased for α ≤ 2.
* **α ≤ 2 carries no resolvable accuracy penalty.** The rise from 0.049 to
  0.056 is inside the across-chain error bars (±0.02). α = 4 is significant
  (2.9× the α = 0 distance) and α = 8 is overwhelming (10×). Resolving the
  small-α region would need more replicate chains than 4.
* **KS is blind to the α = 4 case**: it reads 0.0286, slightly *below* α = 0,
  while W₁(radius) reads 2.9× worse. The displaced mass sits near ‖x‖ = R where
  the CDF is already close to 1, so the sup-gap barely moves.
* **Exact 3-D W₂ has little resolving power** below α = 8 (0.272 vs a floor of
  0.262): E[W₂] decays only like n^(−1/d). The 1-D radius and sliced distances
  are the discriminating statistics; W₂ is reported with its floor alongside.

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
| 2 | 0.005 | 160 000 | 1309 |  46 | 1.59% | 0.0059 |

with W₁(radius) falling 0.049 → 0.030 → 0.034 (α = 0) and 0.056 → 0.049 → 0.041
(α = 2), and the W₁(coord) ratio dropping to 0.63 / 0.58-0.62 — i.e. at small h
the coordinate marginals reach the pure-noise floor and carry no detectable
bias at all.

ESS at fixed simulated time is essentially flat while the discrepancy against
an exact rejection sample shrinks with `h` — i.e. the residual error is the
`O(h)` discretisation/projection bias, not the anchoring, which is exact for
any `δ`.
