# Lasso-regularised constrained sampling, anchored Langevin

The paper's Section 3.3 experiments with `lam |x|_1` added to the potential and
the dynamics replaced by anchored Langevin, so the kinked target is still the
invariant law.  `J = 0` is anchored PSGLD; the other two are non-reversible
anchored Langevin with skew reflection.  Accuracy is the paper's: each walker's
own parameter classifies the set, averaged over walkers, with the standard
deviation across walkers.  `iters to band` is the first iteration whose mean test
accuracy is within 0.005 of the exact constrained lasso posterior's.

## centred ball $K_r = \{x : \|x\|_2^2 \leq r\}$, amplitudes tuned by the sweep

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, centred ball, r = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.03125, `eta` = 6e-06, batch = 50, 16000 iterations, 100 walkers, amplitude scales 1 (constant) and 0.1 (state), tilt 1.

Exact reference: train 0.7494, test 0.7752 (halves 0.7753 / 0.7752), acceptance 0.24, |mean| 0.973, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7489 | 0.7749 | 0.0041 | 57 | 0.817 | 0.067 | 0 |
| constant `J_a`, `a` = 1 | 0.7487 | 0.7742 | 0.0040 | 45 | 0.815 | 0.069 | 0 |
| tilted `J_s(x)`, `s` = [1.0], `c` = 1 | 0.7477 | 0.7735 | 0.0041 | 47 | 0.816 | 0.067 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, centred ball, r = 1.41421, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.003286, `eta` = 6e-06, batch = 30, 16000 iterations, 100 walkers, amplitude scales 0.1 (constant) and 0.1 (state), tilt 3.

Exact reference: train 0.7850, test 0.7910 (halves 0.7910 / 0.7910), acceptance 0.23, |mean| 1.409, 1/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7839 | 0.7861 | 0.0016 | 11 | 0.706 | 0.319 | 0 |
| constant `J_a`, `a` = 0.2 | 0.7839 | 0.7866 | 0.0016 | 11 | 0.701 | 0.317 | 0 |
| tilted `J_s(x)`, `s` = [0.5, 0.5, 0.5], `c` = 3 | 0.7832 | 0.7844 | 0.0019 | 11 | 0.761 | 0.331 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, centred ball, r = 1.41421, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.07013, `eta` = 1e-04, batch = 30, 1500 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 3.

Exact reference: train 0.7808, test 0.7530 (halves 0.7529 / 0.7532), acceptance 0.25, |mean| 1.321, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7797 | 0.7538 | 0.0131 | 486 | 0.311 | 0.036 | 0 |
| constant `J_a`, `a` = 0.9 | 0.7784 | 0.7550 | 0.0134 | 144 | 0.349 | 0.078 | 0 |
| tilted `J_s(x)`, `s` = [0.6, 2.1, 0.6], `c` = 3 | 0.7799 | 0.7560 | 0.0154 | 223 | 0.332 | 0.051 | 0 |

## smoothed $\ell_p$ sublevel set $\{x : g(x) \leq \lambda\}$, amplitudes tuned by the sweep

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, smoothed $\ell_p$ sublevel set, p = 4, $\varepsilon$ = 0.2, $\lambda$ = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.03125, `eta` = 6e-06, batch = 50, 16000 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.1 (state), tilt 2.

Exact reference: train 0.7497, test 0.7755 (halves 0.7755 / 0.7755), acceptance 0.24, |mean| 1.053, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7499 | 0.7757 | 0.0037 | 64 | 0.824 | 0.014 | 0 |
| constant `J_a`, `a` = 0.3 | 0.7497 | 0.7757 | 0.0039 | 49 | 0.825 | 0.015 | 0 |
| tilted `J_g(x)`, `s` = [1.0], `c` = 2 | 0.7491 | 0.7740 | 0.0043 | 53 | 0.822 | 0.015 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.2, $\lambda$ = 4, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.003286, `eta` = 6e-06, batch = 30, 16000 iterations, 100 walkers, amplitude scales 0.1 (constant) and 0.03 (state), tilt 3.

Exact reference: train 0.7837, test 0.7912 (halves 0.7912 / 0.7912), acceptance 0.23, |mean| 1.751, 4/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7835 | 0.7918 | 0.0010 | 12 | 0.636 | 0.001 | 0 |
| constant `J_a`, `a` = 0.2 | 0.7836 | 0.7919 | 0.0010 | 8 | 0.635 | 0.001 | 0 |
| tilted `J_g(x)`, `s` = [0.15, 0.15, 0.15], `c` = 3 | 0.7836 | 0.7919 | 0.0009 | 12 | 0.643 | 0.001 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.18, $\lambda$ = 4, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.07013, `eta` = 1e-04, batch = 30, 2000 iterations, 100 walkers, amplitude scales 1 (constant) and 0.3 (state), tilt 3.

Exact reference: train 0.7819, test 0.7566 (halves 0.7564 / 0.7567), acceptance 0.23, |mean| 1.471, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7815 | 0.7543 | 0.0145 | 839 | 0.327 | 0.000 | 0 |
| constant `J_a`, `a` = 2 | 0.7829 | 0.7629 | 0.0143 | 168 | 0.378 | 0.011 | 2 |
| tilted `J_g(x)`, `s` = [0.6, 2.1, 0.6], `c` = 3 | 0.7814 | 0.7551 | 0.0132 | 371 | 0.354 | 0.000 | 0 |

