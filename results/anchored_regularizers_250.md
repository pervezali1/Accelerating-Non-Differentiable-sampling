# Three regularizers, 250 iterations

Ratios above 1 mean the skew field is closer to the exact constrained posterior
than `J = 0` at the end of the budget.  `gap` is the accuracy gap, `error` the
whitened error of the running posterior mean; amplitudes are selected by
`select_anchored.py --objective gap`.  A cell marked `(bias traded)` is one where
no amplitude met the error guard, so the best gap is reported instead.

| penalty | problem | set | field | kappa | c | gap ratio | error ratio |
|---|---|---|---|---|---|---|---|
| l1 | Titanic | ball | constant `J_a` | 0.3 | 0 | 5.26x +/- 1.47 | 1.84x +/- 0.04 |
| l1 | Titanic | ball | state-dependent `J_s` / `J_g` | 0.3 | 3 | 4.43x +/- 1.65 | 1.64x +/- 0.04 |
| l1 | Titanic | sublevel | constant `J_a` | 0.3 | 0 | 6.77x +/- 2.54 | 1.14x +/- 0.03 |
| l1 | Titanic | sublevel | state-dependent `J_s` / `J_g` | 0.3 | 3 | 1.41x +/- 0.21 | 1.64x +/- 0.04 |
| l1 | MAGIC | ball | constant `J_a` | 0.3 | 0 | 1.51x +/- 0.47 | 3.09x +/- 0.28 |
| l1 | MAGIC | ball | state-dependent `J_s` / `J_g` | 0.3 | 1 | 1.14x +/- 0.46 | 2.39x +/- 0.17 |
| l1 | MAGIC | sublevel | constant `J_a` | 0.3 | 0 | 1.30x +/- 0.40 | 1.11x +/- 0.06 |
| l1 | MAGIC | sublevel | state-dependent `J_s` / `J_g` | 0.03 | 3 | 1.27x +/- 0.17 | 1.02x +/- 0.03 |
| l1 | Synthetic | ball | constant `J_a` | 3 | 0 | 18.79x +/- 4.85 | 2.19x +/- 0.09 |
| l1 | Synthetic | ball | state-dependent `J_s` / `J_g` | 0.1 | 3 | 1.19x +/- 0.07 | 1.15x +/- 0.03 |
| l1 | Synthetic | sublevel | constant `J_a` | 3 | 0 | 20.92x +/- 5.71 | 1.51x +/- 0.04 |
| l1 | Synthetic | sublevel | state-dependent `J_s` / `J_g` | 0.3 | 0 | 1.03x +/- 0.07 | 1.02x +/- 0.02 |
| group | Titanic | ball | constant `J_a` | 0.3 | 0 | 5.52x +/- 1.67 | 2.25x +/- 0.06 |
| group | Titanic | ball | state-dependent `J_s` / `J_g` | 0.3 | 3 | 3.25x +/- 0.80 | 1.78x +/- 0.05 |
| group | Titanic | sublevel | constant `J_a` | 0.3 | 0 | 6.94x +/- 2.38 | 1.20x +/- 0.03 |
| group | Titanic | sublevel | state-dependent `J_s` / `J_g` | 0.3 | 3 | 1.38x +/- 0.22 | 1.60x +/- 0.05 |
| group | MAGIC | ball | constant `J_a` | 0.3 | 0 | 1.72x +/- 1.63 | 2.34x +/- 0.26 |
| group | MAGIC | ball | state-dependent `J_s` / `J_g` | 0.3 | 0 | 2.42x +/- 2.01 | 1.12x +/- 0.13 |
| group | MAGIC | sublevel | constant `J_a` | 0.3 | 0 | 2.21x +/- 0.83 | 1.04x +/- 0.08 |
| group | MAGIC | sublevel | state-dependent `J_s` / `J_g` | 0.3 | 0 | 0.47x +/- 0.27 | 1.06x +/- 0.06 |
| group | Synthetic | ball | constant `J_a` | 1 | 0 | 1.48x +/- 0.06 | 1.17x +/- 0.02 |
| group | Synthetic | ball | state-dependent `J_s` / `J_g` | 0.3 | 1 | 1.02x +/- 0.03 | 1.05x +/- 0.01 |
| group | Synthetic | sublevel | constant `J_a` | 1 | 0 | 1.32x +/- 0.15 | 1.03x +/- 0.03 |
| group | Synthetic | sublevel | state-dependent `J_s` / `J_g` | 0.3 | 0 | 1.02x +/- 0.14 | 1.01x +/- 0.03 |
| tv | Titanic | ball | constant `J_a` | 0.3 | 0 | 5.10x +/- 2.03 | 1.77x +/- 0.04 |
| tv | Titanic | ball | state-dependent `J_s` / `J_g` | 0.3 | 1 | 3.50x +/- 0.83 | 1.24x +/- 0.03 |
| tv | Titanic | sublevel | constant `J_a` | 0.3 | 0 | 6.01x +/- 1.29 | 1.13x +/- 0.02 |
| tv | Titanic | sublevel | state-dependent `J_s` / `J_g` | 0.3 | 3 | 1.59x +/- 0.45 | 1.74x +/- 0.08 |
| tv | MAGIC | ball | constant `J_a` | 0.3 | 0 | 0.79x +/- 0.46 | 1.74x +/- 0.22 |
| tv | MAGIC | ball | state-dependent `J_s` / `J_g` | 0.3 | 0 | 1.04x +/- 0.65 | 1.04x +/- 0.11 |
| tv | MAGIC | sublevel | constant `J_a` | 0.3 | 0 | 0.40x +/- 0.22 | 0.97x +/- 0.07 |
| tv | MAGIC | sublevel | state-dependent `J_s` / `J_g` | 0.3 | 1 | 0.78x +/- 0.62 | 1.30x +/- 0.21 |
| tv | Synthetic | ball | constant `J_a` | 1 | 0 | 1.61x +/- 0.09 | 1.13x +/- 0.02 |
| tv | Synthetic | ball | state-dependent `J_s` / `J_g` | 0.3 | 1 | 1.06x +/- 0.04 | 1.04x +/- 0.02 |
| tv | Synthetic | sublevel | constant `J_a` | 1 | 0 | 1.50x +/- 0.18 | 0.99x +/- 0.03 |
| tv | Synthetic | sublevel | state-dependent `J_s` / `J_g` | 0.3 | 0 | 1.00x +/- 0.13 | 1.00x +/- 0.03 |
