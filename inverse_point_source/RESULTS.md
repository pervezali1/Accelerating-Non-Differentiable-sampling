# Two-source inverse point-source problem: what actually limits the reconstruction

Everything below is measured, not asserted. The scripts that produced each number
are described inline so they can be re-run.

Setup under study: `k = 5`, `λ = 1.2566`, two point sources of equal strength
`α = 1 − 2i` in `[−1, 1]²`, separation drawn uniformly from `[λ/4, λ/2] =
[0.3142, 0.6283]`, Cauchy data `(u, ∂u/∂n)` at 256 receivers on a circle of
radius `R = 50`, reconstructed as an indicator on a 64×64 grid
(spacing `0.03175`) through a fixed Gaussian-RBF trunk.

---

## 1. The forward operator is correct

Three independent checks on `Φ = (i/4)H₀⁽¹⁾(kr)`,
`∂Φ/∂n = (−ik/4)H₁⁽¹⁾(kr)·∂r/∂n`:

| check | result |
|---|---|
| analytic `∂u/∂n` vs central difference along the outward normal | max rel. err `8.8e-10` |
| Helmholtz residual `\|Δu + k²u\| / \|k²u\|` away from the sources | `1.9e-8` |
| Sommerfeld: `√R·\|∂u/∂r − iku\|` at `R = 50, 200, 800` | `4.0e-3, 1.0e-3, 2.5e-4` |

The Sommerfeld quantity falls by ~4× per 4× in `R`, i.e. `\|∂u/∂r − iku\| ~ R^{−3/2}`,
as it should for an outgoing solution. No bug in the physics.

---

## 2. The RBF trunk is the dominant error, not the branch network

The branch output is squared, so the predicted indicator is a **non-negative**
combination of the trunk's Gaussians. `scipy.optimize.nnls` finds the best such
combination for a given target, which is a hard lower bound on what *any* branch
network paired with that trunk can achieve. Averaged over 12 random two-source
configurations, under the same weighted MSE (`w = 1 + 20·target`):

| trunk | `P` | `σ_rbf` | best achievable loss | peak height |
|---|---|---|---|---|
| predict zero everywhere | — | — | `8.65e-02` | 0.00 |
| **`h = λ/8`, `s = 0.15` (original)** | **169** | **0.0806** | **`1.24e-02`** | **0.763** |
| `h = λ/8`, `s = 0.50` | 169 | 0.1334 | `1.75e-02` | 0.607 |
| `h = λ/12`, `s = 0.15` | 400 | 0.0538 | `2.39e-03` | 0.932 |
| `h = λ/16`, `s = 0.15` | 676 | 0.0403 | `5.19e-04` | 0.983 |
| **`h = λ/16`, `s = 0.30`** | **676** | **0.0506** | **`2.60e-04`** | **0.971** |
| `h = λ/20`, `s = 0.15` | 1024 | 0.0323 | `1.97e-04` | 0.997 |

The original notebook trains to `2.14e-02`. Its floor is `1.24e-02`. So **58 % of
the reported loss is the trunk** and cannot be trained away. The floor also caps
the peak height at 0.763, which is most of why the trained model tops out at 0.59
rather than 1.0.

Target width is `σ_tgt = 2·grid_h = 0.0635`, and the original trunk's Gaussians
are `σ_rbf = 0.0806` — *wider than the thing they are being asked to fit*. A
non-negative sum of Gaussians of width 0.0806 is never narrower than 0.0806.

**Two design rules** follow, both violated by the original settings:

1. `σ_rbf ≤ σ_tgt`, i.e. `h ≤ σ_tgt·√(−2 ln s)`. Original: `0.0806 > 0.0635`.
2. `h ≤ d_min/2`. Original: `h = 0.1571`, `d_min/2 = 0.1571` — exactly on the
   boundary, so two sources at the minimum separation are two lattice pitches
   apart and their representable footprints touch.

Widening the *target* instead of narrowing the trunk also works and is cheaper
(`σ_tgt = 0.12` with the original trunk gives a floor of `4.86e-03`), but it
throws away localisation sharpness, so narrowing the trunk is the better trade.

### Contrast between two lobes 0.30 apart, at the floor

| trunk | peak | saddle | valley depth |
|---|---|---|---|
| `h = λ/8`, `s = 0.15` | 0.793 | 0.242 | 69.5 % |
| `h = λ/12`, `s = 0.15` | 0.842 | 0.203 | 75.9 % |
| `h = λ/16`, `s = 0.30` | 0.895 | 0.105 | 88.3 % |

---

## 3. The branch input is unnormalised, and it costs a third of the loss

Measured over 400 random configurations:

| block | mean | std |
|---|---|---|
| `Re u` | 0.00041 | 0.02912 |
| `Im u` | 0.00519 | 0.02867 |
| `Re ∂u/∂n` | −0.02593 | 0.14336 |
| `Im ∂u/∂n` | 0.00200 | 0.14562 |

Overall input std is `0.105`. With PyTorch's default `Linear` init
(`U(±1/√fan_in)`, std `0.018`) the first-layer pre-activation std is
`√1024 · 0.018 · 0.105 = 0.061`. Both `tanh` layers therefore start deep in
their linear region, and after the `b²` positivity map the network's output
starts orders of magnitude below the `O(1)` target. Standardising the input
raises the pre-activation std to `0.577`, which is the regime the
initialisation was designed for.

The `∂u/∂n` block is also 5× larger than the `u` block, so at initialisation the
network effectively ignores `u`.

---

## 4. At `R = 50` the Cauchy data is redundant

For an outgoing field, `∂u/∂n → ik·u` as `kR → ∞`. Here `kR = 250`. Fitting the
single best complex scalar `c` and measuring the residual:

| `R` | `kR` | `‖∂u/∂n − c·u‖ / ‖∂u/∂n‖` |
|---|---|---|
| 50 | 250 | `3.0e-05` |
| 10 | 50 | `6.2e-04` |
| 3 | 15 | `1.1e-02` |
| 1.5 | 7.5 | `5.5e-02` |

So the 1024-dimensional branch input contains ~512 independent numbers, and the
first `Linear` layer carries twice the parameters it needs. This costs nothing on
noiseless data, but if you want `∂u/∂n` to be a genuinely second measurement —
which is what would help under noise — the receivers have to come in.

---

## 5. The measurement is *not* what limits the resolution

`σ_min(J)/‖b‖` is the smallest relative change in the data produced by a unit
displacement of a source; dividing a noise level by it gives the position error
that noise level supports.

| separation | `σ_min(J)/‖b‖` | error @ 1 % noise | @ 0.1 % noise |
|---|---|---|---|
| `λ/2 = 0.6283` | 1.375 | 0.0073 | 0.00073 |
| `λ/4 = 0.3142` | 1.087 | 0.0092 | 0.00092 |
| `λ/8 = 0.1571` | 0.504 | 0.0199 | 0.0020 |
| `λ/16 = 0.0785` | 0.247 | 0.0405 | 0.0041 |

Grid spacing for comparison: `0.0317`. The condition number of the 4-parameter
Jacobian is 2.6 at every radius tested.

At the training separation `λ/4`, **1 % noise already supports a localisation
error of 0.009 — about three times finer than one grid cell**, and roughly four
times better than the current pipeline delivers. The accuracy that is missing is
not information the measurement lacks. It is being lost in the trunk, in the
training, and in the peak picking.

---

## 6. Peak detection

Original recipe: `I >= 0.60·I.max()` combined with
`I == maximum_filter(I, size=5)`. Three problems:

* **The threshold is relative to the global maximum.** The model undershoots the
  peak and does not undershoot both lobes equally. A weaker lobe at 0.35 against
  a stronger one at 0.59 sits at ratio 0.59 — just under the cut — and vanishes.
* **`size=5` is a magic number in pixels.** Its suppression radius is 2 cells =
  0.063, far below the 0.314 separation being resolved, so it suppresses nothing
  and shoulder ripples can register as sources. The radius should come from the
  physics: `≈ 0.4·d_min` in cells.
* **`I == maximum_filter(I)` mishandles flat plateaus** — every cell of a flat
  summit passes, so one lobe reports several coincident "sources".

Replacement: topographic **prominence** (the drop from a summit to the highest
saddle connecting it to a taller peak), which is scale-free and therefore immune
to the amplitude undershoot; plateaus collapsed by connected-component labelling;
non-maximum suppression in physical units; sub-grid parabolic refinement of the
summit.

Benchmark, 200 synthetic fields with unequal lobe strengths and varying blur:

| | correct source count | mean localisation error |
|---|---|---|
| threshold 0.60, size 5 | 85.5 % | 0.0402 |
| prominence + NMS + sub-grid | **95.5 %** | 0.0390 |

Sub-grid refinement in isolation, on *sharp* fields: **0.0131 → 0.0055**. On
heavily blurred fields it only moves 0.0400 → 0.0385, because there the error is
dominated by blur pulling the two summits toward each other rather than by grid
quantisation. Fix the trunk first, then the refinement pays.

For context on why this matters: the 64×64 grid `linspace(−1, 1, 64)` does not
contain the point `0`. A source at the origin can only be reported at
`±0.0159`, which is exactly the "error" the original notebook showed for the
`(0, 0)` demo source.

Rather than guessing the cutoff, the notebook sweeps `prom_frac` on the
validation split and keeps whatever maximises the correct-count rate.

---

## 7. Rotation augmentation is exact

The receivers are at 256 equispaced angles on a circle centred at the origin, so
rotating both sources by `2πn/256` maps the Cauchy data to a circular shift of
itself:

```
u(rotated sources) == np.roll(u, n)      max rel. err 5.6e-14
∂u/∂n(rotated)     == np.roll(∂u/∂n, n)  max rel. err 5.6e-14
```

Exact to machine precision, no interpolation. Each training sample stands in for
256, at the cost of one `torch.roll` per batch.

---

## 8. Training-loop bugs

* `CosineAnnealingLR(T_max=300)` with `num_epochs=100`: the learning rate
  traversed only a third of the cosine, so the final model was still training at
  ~0.75·LR when the loop stopped and never annealed. (The cross-validation cell
  had this right; only the final training run did not.)
* `s = 0.15` (RBF overlap, Cell 3) is overwritten by `s = source_locations[i,:N]`
  (Cell 4). Harmless in a top-to-bottom run, silently wrong on any re-execution
  of Cell 3 afterwards.
* `weight_decay` on `torch.optim.Adam` is L2-in-the-gradient, which the adaptive
  per-parameter scaling then rescales. `AdamW` applies the decoupled decay that
  `weight_decay` is normally taken to mean.
* The "completely new test data" demo uses separation `0.30`, below the training
  `d_min = 0.3142`. That figure is an extrapolation.
* `I_true` is 60000 × 4096 float32 = **983 MB**, plus 246 MB for `X_all`.


---

## 9. Ablations

All runs use the same 8 000-sample dataset (6 400 train / 1 600 val), the same
seed, batch 64, `AdamW(1e-3, wd 5e-5)`, `CosineAnnealingLR`, 40 epochs, and the
same weighted MSE for reporting. The baseline reproduces the original notebook's
behaviour closely (`2.23e-02` here against `2.14e-02` in the original run on
48 000 samples), and two independently generated datasets gave `1.5802e-02` and
`1.5747e-02` for the same configuration — a 0.3 % spread, so the differences
below are well above run-to-run noise.

### 9.1 Trunk `h = λ/8`, `s = 0.15` (representational floor `1.24e-02`)

| run | best val loss | mean peak |
|---|---|---|
| A — original configuration | `2.23e-02` | 0.647 |
| B — `T_max` fixed to `NUM_EPOCHS` | `2.35e-02` | 0.621 |
| C — B + input standardisation | `1.58e-02` | 0.719 |
| **D1 — C + GELU instead of tanh** | **`1.42e-02`** | 0.734 |

Cumulative on the original trunk: **`2.23e-02` → `1.42e-02`, −36 %**, from two
changes that cost nothing at runtime. At `1.42e-02` the network is only 14 %
above this trunk's `1.24e-02` floor — there is almost nothing left to win
without changing the trunk.

**B is not an improvement at this budget, and that is expected.** With
`T_max = 300` and only 40 (or 100) epochs the learning rate stays high, which
helps a run nowhere near converged. The fix matters once you train to
convergence with early stopping; it is a correctness fix, not a free win, and it
is honest to report it as slightly worse in a truncated comparison.

### 9.2 Positivity map and activation

Same trunk, same budget, input standardisation on throughout.

| activation + positivity | best val loss | mean peak |
|---|---|---|
| **GELU + `b²`** | **`1.4150e-02`** | 0.734 |
| GELU + `softplus(b−4)` | `1.5090e-02` | **0.766** |
| tanh + `b²` (the paper's combination) | `1.5747e-02` | 0.719 |
| tanh + softplus, last-layer bias −4 | `1.6789e-02` | 0.718 |
| tanh + `softplus(b−4)` | `1.6794e-02` | 0.726 |
| tanh + `softplus(b)` | `1.7862e-02` | 0.716 |
| GELU + `softplus(b)` unshifted | `8.7072e-02` | **0.000** |

Reading of this table:

* **`b²` wins.** The theoretical objection to it — `d(b²)/db = 2b` vanishes
  exactly where `b = 0`, so units sitting near zero get no gradient — is real,
  but it only bites while the inputs are unnormalised and nearly every unit *is*
  near zero. Once standardised, `b²` is the best map measured.
* **`softplus(0) = 0.693` is the problem with the naive swap**: at init all `P`
  coefficients are ≈ 0.7 and the field is a large positive constant over the
  whole domain. With tanh this merely costs accuracy (`1.79e-02`). With GELU it
  is fatal — the optimiser drives the pre-activations very negative, softplus
  saturates at 0 along with its derivative, and the model parks at exactly the
  predict-zero loss with a mean peak of 0.000.
* **Shifting the argument by 4 fixes it**, and two independent routes to the same
  shift agree: `softplus(b−4)` gives `1.6794e-02` and initialising the last-layer
  bias to −4 gives `1.6789e-02`. For the GELU pairing the shift is worth `8.71e-02
  → 1.51e-02`, a 5.8× difference from one constant.
* **One inversion worth noting:** softplus produces the *highest peak* (0.766)
  while scoring worse on MSE. If peak contrast between two merged lobes matters
  more to you than fit, that trade is worth revisiting.
