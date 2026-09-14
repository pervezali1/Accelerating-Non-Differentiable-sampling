# Lasso-regularised constrained sampling, anchored Langevin

The paper's Section 3.3 experiments with `lam |x|_1` added to the potential and
the dynamics replaced by anchored Langevin, so the kinked target is still the
invariant law.  `J = 0` is anchored PSGLD; the other two are non-reversible
anchored Langevin with skew reflection.  Accuracy is the paper's: each walker's
own parameter classifies the set, averaged over walkers, with the standard
deviation across walkers.  `iters to band` is the first iteration whose mean test
accuracy is within 0.005 of the exact constrained lasso posterior's.

## centred ball $K_r = \{x : \|x\|_2^2 \leq r\}$, a 250-iteration budget, amplitudes tuned for it

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, centred ball, r = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.03125, `eta` = 6e-06, batch = 50, 250 iterations, 100 walkers, amplitude scales 3 (constant) and 0.1 (state), tilt 3.

Exact reference: train 0.7494, test 0.7752 (halves 0.7753 / 0.7752), acceptance 0.24, |mean| 0.973, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7152 | 0.7299 | 0.0437 | never | 0.836 | 0.001 | 0 |
| constant `J_a`, `a` = 3 | 0.7331 | 0.7758 | 0.0089 | 211 | 0.872 | 0.100 | 0 |
| tilted `J_s(x)`, `s` = [1.0], `c` = 3 | 0.7097 | 0.7358 | 0.0366 | never | 0.852 | 0.001 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, centred ball, r = 1.41421, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.003286, `eta` = 6e-06, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 1.

Exact reference: train 0.7850, test 0.7910 (halves 0.7910 / 0.7910), acceptance 0.23, |mean| 1.409, 1/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7822 | 0.7878 | 0.0024 | 151 | 0.808 | 0.001 | 0 |
| constant `J_a`, `a` = 0.6 | 0.7840 | 0.7887 | 0.0025 | 134 | 0.806 | 0.088 | 0 |
| tilted `J_s(x)`, `s` = [1.5, 1.5, 1.5], `c` = 1 | 0.7821 | 0.7886 | 0.0019 | 182 | 0.847 | 0.038 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, centred ball, r = 1.41421, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.07013, `eta` = 1e-04, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 3.

Exact reference: train 0.7808, test 0.7530 (halves 0.7529 / 0.7532), acceptance 0.25, |mean| 1.321, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7622 | 0.7350 | 0.0201 | never | 0.280 | 0.000 | 0 |
| constant `J_a`, `a` = 0.9 | 0.7745 | 0.7503 | 0.0163 | 144 | 0.383 | 0.078 | 0 |
| tilted `J_s(x)`, `s` = [0.6, 2.1, 0.6], `c` = 3 | 0.7689 | 0.7446 | 0.0163 | 223 | 0.352 | 0.005 | 0 |

## smoothed $\ell_p$ sublevel set $\{x : g(x) \leq \lambda\}$, a 250-iteration budget, amplitudes tuned for it

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, smoothed $\ell_p$ sublevel set, p = 4, $\varepsilon$ = 0.2, $\lambda$ = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.03125, `eta` = 6e-06, batch = 50, 250 iterations, 100 walkers, amplitude scales 3 (constant) and 0.3 (state), tilt 0.

Exact reference: train 0.7497, test 0.7755 (halves 0.7755 / 0.7755), acceptance 0.24, |mean| 1.053, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7182 | 0.7292 | 0.0475 | never | 0.811 | 0.000 | 0 |
| constant `J_a`, `a` = 3 | 0.7307 | 0.7739 | 0.0104 | 183 | 0.888 | 0.079 | 0 |
| state-dependent `J_g(x)`, `s` = [3.0] | 0.7187 | 0.7329 | 0.0468 | never | 0.834 | 0.000 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.2, $\lambda$ = 4, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.003286, `eta` = 6e-06, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.03 (state), tilt 3.

Exact reference: train 0.7837, test 0.7912 (halves 0.7912 / 0.7912), acceptance 0.23, |mean| 1.751, 4/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7815 | 0.7872 | 0.0024 | 192 | 0.796 | 0.000 | 0 |
| constant `J_a`, `a` = 0.6 | 0.7835 | 0.7885 | 0.0023 | 134 | 0.813 | 0.000 | 0 |
| tilted `J_g(x)`, `s` = [0.15, 0.15, 0.15], `c` = 3 | 0.7839 | 0.7890 | 0.0022 | 183 | 0.816 | 0.000 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.18, $\lambda$ = 4, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.07013, `eta` = 1e-04, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 3.

Exact reference: train 0.7819, test 0.7566 (halves 0.7564 / 0.7567), acceptance 0.23, |mean| 1.471, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7652 | 0.7413 | 0.0221 | never | 0.277 | 0.000 | 0 |
| constant `J_a`, `a` = 0.6 | 0.7810 | 0.7631 | 0.0144 | 173 | 0.361 | 0.000 | 0 |
| tilted `J_g(x)`, `s` = [0.6, 2.1, 0.6], `c` = 3 | 0.7677 | 0.7467 | 0.0189 | never | 0.364 | 0.000 | 0 |

