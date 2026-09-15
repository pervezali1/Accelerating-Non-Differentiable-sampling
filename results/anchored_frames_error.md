# The frame the state-dependent field is read in, 250 iterations

Which coordinates share a `3`-block of `J_s` is free: conjugating by an
orthogonal `Q` keeps the field skew and divergence free and keeps `J(x) x = 0`,
so it is the paper's construction in another frame.  `Q` also has to keep the
constraint set invariant, which on the smoothed `l_p` set means signed
permutations -- so that is the group searched.  `columns` is the paper's own
order; `spectral` is the signed permutation maximising the slowest rate of
`(I + J) H` at the constrained MAP (`nds.constrained.spectral_frame`).

`rate` is that predicted ratio, computed with no simulation at all; `gap` and
`error` are measured, ratios of `J = 0` over the field, so above 1 the field
wins.  Amplitudes are selected by the rule in `select_anchored.py` against the
gap.  Uncertainties are standard errors over the walker seeds.

| problem | set | penalty | eta | field | frame | kappa | c | rate | gap ratio | error ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| MAGIC | ball | group | 1.56e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| MAGIC | ball | group | 1.56e-06 | `J_a` | -- | 1 | 0 | -- | 1.99x +/- 0.13 | 3.30x +/- 0.16 |
| MAGIC | ball | group | 1.56e-06 | `J_s` / `J_g` | columns | 1 | 0 | -- | 1.11x +/- 0.04 | 1.05x +/- 0.01 |
| MAGIC | ball | group | 1.56e-06 | `J_s` / `J_g` | spectral | 0.3 | 3 | 2.68x | 1.27x +/- 0.10 | 1.53x +/- 0.03 |
| MAGIC | ball | group | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| MAGIC | ball | group | 6.25e-06 | `J_a` | -- | 0.3 | 0 | -- | 1.72x +/- 1.63 | 2.34x +/- 0.26 |
| MAGIC | ball | group | 6.25e-06 | `J_s` / `J_g` | columns | 0.3 | 0 | -- | 2.42x +/- 2.01 | 1.12x +/- 0.13 |
| MAGIC | ball | l1 | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| MAGIC | ball | l1 | 6.25e-06 | `J_a` | -- | 0.3 | 0 | -- | 1.51x +/- 0.47 | 3.09x +/- 0.28 |
| MAGIC | ball | l1 | 6.25e-06 | `J_s` / `J_g` | columns | 0.3 | 1 | -- | 1.14x +/- 0.46 | 2.39x +/- 0.17 |
| MAGIC | ball | tv | 1.56e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| MAGIC | ball | tv | 1.56e-06 | `J_a` | -- | 1 | 0 | 2.74x | 3.72x +/- 0.78 | 2.55x +/- 0.14 |
| MAGIC | ball | tv | 1.56e-06 | `J_s` / `J_g` | columns | 1 | 0 | 2.08x | 1.31x +/- 0.16 | 1.03x +/- 0.01 |
| MAGIC | ball | tv | 1.56e-06 | `J_s` / `J_g` | spectral | 0.3 | 1 | 3.07x | 1.52x +/- 0.04 | 1.04x +/- 0.01 |
| MAGIC | ball | tv | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| MAGIC | ball | tv | 6.25e-06 | `J_a` | -- | 0.3 | 0 | -- | 0.79x +/- 0.46 | 1.74x +/- 0.22 |
| MAGIC | ball | tv | 6.25e-06 | `J_s` / `J_g` | columns | 0.3 | 0 | -- | 1.04x +/- 0.65 | 1.04x +/- 0.11 |
| MAGIC | sublevel | group | 1.56e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| MAGIC | sublevel | group | 1.56e-06 | `J_a` | -- | 0.3 | 0 | -- | 0.72x +/- 0.06 | 1.00x +/- 0.02 |
| MAGIC | sublevel | group | 1.56e-06 | `J_s` / `J_g` | columns | 0.3 | 0 | -- | 0.91x +/- 0.07 | 1.01x +/- 0.02 |
| MAGIC | sublevel | group | 1.56e-06 | `J_s` / `J_g` | spectral | 0.3 | 0 | 2.37x | 1.00x +/- 0.07 | 1.03x +/- 0.02 |
| MAGIC | sublevel | group | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| MAGIC | sublevel | group | 6.25e-06 | `J_a` | -- | 0.3 | 0 | -- | 2.21x +/- 0.83 | 1.04x +/- 0.08 |
| MAGIC | sublevel | group | 6.25e-06 | `J_s` / `J_g` | columns | 0.3 | 0 | -- | 0.47x +/- 0.27 | 1.06x +/- 0.06 |
| MAGIC | sublevel | l1 | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| MAGIC | sublevel | l1 | 6.25e-06 | `J_a` | -- | 0.3 | 0 | -- | 1.30x +/- 0.40 | 1.11x +/- 0.06 |
| MAGIC | sublevel | l1 | 6.25e-06 | `J_s` / `J_g` | columns | 0.03 | 3 | -- | 1.27x +/- 0.17 | 1.02x +/- 0.03 |
| MAGIC | sublevel | tv | 1.56e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| MAGIC | sublevel | tv | 1.56e-06 | `J_a` | -- | 0.3 | 0 | 1.75x | 0.81x +/- 0.09 | 1.00x +/- 0.01 |
| MAGIC | sublevel | tv | 1.56e-06 | `J_s` / `J_g` | columns | 0.3 | 0 | 1.92x | 0.89x +/- 0.08 | 1.01x +/- 0.02 |
| MAGIC | sublevel | tv | 1.56e-06 | `J_s` / `J_g` | spectral | 1 | 0 | 2.66x | 1.17x +/- 0.26 | 1.10x +/- 0.03 |
| MAGIC | sublevel | tv | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| MAGIC | sublevel | tv | 6.25e-06 | `J_a` | -- | 0.3 | 0 | -- | 0.40x +/- 0.22 | 0.97x +/- 0.07 |
| MAGIC | sublevel | tv | 6.25e-06 | `J_s` / `J_g` | columns | 0.3 | 1 | -- | 0.78x +/- 0.62 | 1.30x +/- 0.21 |
| Synthetic | ball | group | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Synthetic | ball | group | 6.25e-06 | `J_a` | -- | 1 | 0 | -- | 1.48x +/- 0.06 | 1.17x +/- 0.02 |
| Synthetic | ball | group | 6.25e-06 | `J_s` / `J_g` | columns | 0.3 | 1 | -- | 1.02x +/- 0.03 | 1.05x +/- 0.01 |
| Synthetic | ball | l1 | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Synthetic | ball | l1 | 6.25e-06 | `J_a` | -- | 3 | 0 | -- | 18.79x +/- 4.85 | 2.19x +/- 0.09 |
| Synthetic | ball | l1 | 6.25e-06 | `J_s` / `J_g` | columns | 0.1 | 3 | -- | 1.19x +/- 0.07 | 1.15x +/- 0.03 |
| Synthetic | ball | l1 | 2.5e-05 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Synthetic | ball | l1 | 2.5e-05 | `J_a` | -- | 0.3 | 0 | -- | 0.46x +/- 0.28 | 1.28x +/- 0.06 |
| Synthetic | ball | l1 | 2.5e-05 | `J_s` / `J_g` | columns | 0.3 | 0 | -- | 0.25x +/- 0.15 | 1.01x +/- 0.03 |
| Synthetic | ball | tv | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Synthetic | ball | tv | 6.25e-06 | `J_a` | -- | 1 | 0 | -- | 1.61x +/- 0.09 | 1.13x +/- 0.02 |
| Synthetic | ball | tv | 6.25e-06 | `J_s` / `J_g` | columns | 0.3 | 1 | -- | 1.06x +/- 0.04 | 1.04x +/- 0.02 |
| Synthetic | sublevel | group | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Synthetic | sublevel | group | 6.25e-06 | `J_a` | -- | 1 | 0 | -- | 1.32x +/- 0.15 | 1.03x +/- 0.03 |
| Synthetic | sublevel | group | 6.25e-06 | `J_s` / `J_g` | columns | 0.3 | 0 | -- | 1.02x +/- 0.14 | 1.01x +/- 0.03 |
| Synthetic | sublevel | l1 | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Synthetic | sublevel | l1 | 6.25e-06 | `J_a` | -- | 3 | 0 | -- | 20.92x +/- 5.71 | 1.51x +/- 0.04 |
| Synthetic | sublevel | l1 | 6.25e-06 | `J_s` / `J_g` | columns | 0.3 | 0 | -- | 1.03x +/- 0.07 | 1.02x +/- 0.02 |
| Synthetic | sublevel | l1 | 2.5e-05 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Synthetic | sublevel | l1 | 2.5e-05 | `J_a` | -- | 0.3 | 0 | -- | 1.00x +/- 0.61 | 1.10x +/- 0.09 |
| Synthetic | sublevel | l1 | 2.5e-05 | `J_s` / `J_g` | columns | 0.3 | 0 | -- | 0.20x +/- 0.09 | 1.08x +/- 0.10 |
| Synthetic | sublevel | tv | 6.25e-06 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Synthetic | sublevel | tv | 6.25e-06 | `J_a` | -- | 1 | 0 | -- | 1.50x +/- 0.18 | 0.99x +/- 0.03 |
| Synthetic | sublevel | tv | 6.25e-06 | `J_s` / `J_g` | columns | 0.3 | 0 | -- | 1.00x +/- 0.13 | 1.00x +/- 0.03 |
| Titanic | ball | group | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | ball | group | 0.0001 | `J_s` / `J_g` | spectral | 0.3 | 3 | 3.76x | 1.81x +/- 0.34 | 1.72x +/- 0.04 |
| Titanic | ball | group | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | ball | group | 0.0001 | `J_a` | -- | 0.3 | 0 | -- | 5.52x +/- 1.67 | 2.25x +/- 0.06 |
| Titanic | ball | group | 0.0001 | `J_s` / `J_g` | columns | 0.3 | 3 | -- | 3.25x +/- 0.80 | 1.78x +/- 0.05 |
| Titanic | ball | l1 | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | ball | l1 | 0.0001 | `J_s` / `J_g` | spectral | 0.3 | 1 | 4.04x | 2.64x +/- 1.07 | 1.13x +/- 0.01 |
| Titanic | ball | l1 | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | ball | l1 | 0.0001 | `J_a` | -- | 0.3 | 0 | -- | 5.26x +/- 1.47 | 1.84x +/- 0.04 |
| Titanic | ball | l1 | 0.0001 | `J_s` / `J_g` | columns | 0.3 | 3 | -- | 4.43x +/- 1.65 | 1.64x +/- 0.04 |
| Titanic | ball | tv | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | ball | tv | 0.0001 | `J_s` / `J_g` | spectral | 0.3 | 0 | 2.87x | 1.26x +/- 0.33 | 1.00x +/- 0.02 |
| Titanic | ball | tv | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | ball | tv | 0.0001 | `J_a` | -- | 0.3 | 0 | -- | 5.10x +/- 2.03 | 1.77x +/- 0.04 |
| Titanic | ball | tv | 0.0001 | `J_s` / `J_g` | columns | 0.3 | 1 | -- | 3.50x +/- 0.83 | 1.24x +/- 0.03 |
| Titanic | sublevel | group | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | sublevel | group | 0.0001 | `J_s` / `J_g` | spectral | 1 | 1 | 4.36x | 2.93x +/- 1.40 | 1.40x +/- 0.06 |
| Titanic | sublevel | group | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | sublevel | group | 0.0001 | `J_a` | -- | 0.3 | 0 | -- | 6.94x +/- 2.38 | 1.20x +/- 0.03 |
| Titanic | sublevel | group | 0.0001 | `J_s` / `J_g` | columns | 0.3 | 3 | -- | 1.38x +/- 0.22 | 1.60x +/- 0.05 |
| Titanic | sublevel | l1 | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | sublevel | l1 | 0.0001 | `J_s` / `J_g` | spectral | 0.3 | 1 | 4.83x | 3.07x +/- 1.55 | 1.28x +/- 0.03 |
| Titanic | sublevel | l1 | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | sublevel | l1 | 0.0001 | `J_a` | -- | 0.3 | 0 | -- | 6.77x +/- 2.54 | 1.14x +/- 0.03 |
| Titanic | sublevel | l1 | 0.0001 | `J_s` / `J_g` | columns | 0.3 | 3 | -- | 1.41x +/- 0.21 | 1.64x +/- 0.04 |
| Titanic | sublevel | tv | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | sublevel | tv | 0.0001 | `J_s` / `J_g` | spectral | 0.3 | 3 | 5.07x | 6.83x +/- 1.40 | 1.41x +/- 0.13 |
| Titanic | sublevel | tv | 0.0001 | `J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |
| Titanic | sublevel | tv | 0.0001 | `J_a` | -- | 0.3 | 0 | -- | 6.01x +/- 1.29 | 1.13x +/- 0.02 |
| Titanic | sublevel | tv | 0.0001 | `J_s` / `J_g` | columns | 0.3 | 3 | -- | 1.59x +/- 0.45 | 1.74x +/- 0.08 |
