# Lasso-regularised constrained sampling, anchored Langevin

The paper's Section 3.3 experiments with `lam |x|_1` added to the potential and
the dynamics replaced by anchored Langevin, so the kinked target is still the
invariant law.  `J = 0` is anchored PSGLD; the other two are non-reversible
anchored Langevin with skew reflection.  Accuracy is the paper's: each walker's
own parameter classifies the set, averaged over walkers, with the standard
deviation across walkers.  `iters to band` is the first iteration whose mean test
accuracy is within 0.005 of the exact constrained lasso posterior's.

## centred ball $K_r = \{x : \|x\|_2^2 \leq r\}$, calibrated amplitudes

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, centred ball, r = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.09375, `eta` = 6e-06, batch = 50, 250 iterations, 100 walkers, amplitude scales 1 (constant) and 0.3 (state), tilt 1.

Exact reference: train 0.7492, test 0.7742 (halves 0.7741 / 0.7742), acceptance 0.25, |mean| 0.976, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7201 | 0.7355 | 0.0396 | never | 0.873 | 0.001 | 0 |
| constant `J_a`, `a` = 1 | 0.7150 | 0.7498 | 0.0322 | never | 0.883 | 0.049 | 0 |
| tilted `J_s(x)`, `s` = [3.0], `c` = 1 | 0.7151 | 0.7340 | 0.0434 | never | 0.881 | 0.001 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, centred ball, r = 1.41421, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.009858, `eta` = 6e-06, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 0.

Exact reference: train 0.7801, test 0.7850 (halves 0.7849 / 0.7850), acceptance 0.23, |mean| 1.409, 1/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7796 | 0.7842 | 0.0025 | 60 | 0.957 | 0.021 | 0 |
| constant `J_a`, `a` = 0.6 | 0.7813 | 0.7858 | 0.0025 | 49 | 0.963 | 0.149 | 0 |
| state-dependent `J_s(x)`, `s` = [1.5, 1.5, 1.5] | 0.7811 | 0.7854 | 0.0022 | 48 | 0.958 | 0.030 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, centred ball, r = 1.41421, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.2104, `eta` = 1e-04, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 3.

Exact reference: train 0.7808, test 0.7549 (halves 0.7547 / 0.7550), acceptance 0.25, |mean| 1.334, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7662 | 0.7376 | 0.0170 | never | 0.327 | 0.000 | 0 |
| constant `J_a`, `a` = 0.9 | 0.7755 | 0.7521 | 0.0165 | 143 | 0.434 | 0.105 | 0 |
| tilted `J_s(x)`, `s` = [0.6, 2.1, 0.6], `c` = 3 | 0.7706 | 0.7429 | 0.0182 | 205 | 0.402 | 0.011 | 0 |

## centred ball $K_r = \{x : \|x\|_2^2 \leq r\}$, calibrated amplitudes

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, centred ball, r = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.04688, `eta` = 6e-06, batch = 50, 250 iterations, 100 walkers, amplitude scales 1 (constant) and 0.3 (state), tilt 1.

Exact reference: train 0.7479, test 0.7743 (halves 0.7743 / 0.7744), acceptance 0.23, |mean| 0.968, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7132 | 0.7287 | 0.0456 | never | 0.841 | 0.001 | 0 |
| constant `J_a`, `a` = 1 | 0.7144 | 0.7480 | 0.0344 | never | 0.875 | 0.042 | 0 |
| tilted `J_s(x)`, `s` = [3.0], `c` = 1 | 0.7082 | 0.7262 | 0.0462 | never | 0.843 | 0.001 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, centred ball, r = 1.41421, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.003697, `eta` = 6e-06, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 0.

Exact reference: train 0.7746, test 0.7778 (halves 0.7778 / 0.7778), acceptance 0.23, |mean| 1.405, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7764 | 0.7806 | 0.0030 | 47 | 0.870 | 0.000 | 0 |
| constant `J_a`, `a` = 0.6 | 0.7788 | 0.7830 | 0.0026 | 40 | 0.884 | 0.055 | 0 |
| state-dependent `J_s(x)`, `s` = [1.5, 1.5, 1.5] | 0.7781 | 0.7819 | 0.0026 | 38 | 0.880 | 0.001 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, centred ball, r = 1.41421, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.07889, `eta` = 1e-04, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 1.

Exact reference: train 0.7841, test 0.7547 (halves 0.7546 / 0.7548), acceptance 0.25, |mean| 1.314, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7684 | 0.7393 | 0.0177 | never | 0.330 | 0.000 | 0 |
| constant `J_a`, `a` = 0.9 | 0.7768 | 0.7537 | 0.0161 | 132 | 0.429 | 0.076 | 0 |
| tilted `J_s(x)`, `s` = [0.6, 2.1, 0.6], `c` = 1 | 0.7752 | 0.7478 | 0.0140 | 179 | 0.360 | 0.000 | 0 |

## smoothed $\ell_p$ sublevel set $\{x : g(x) \leq \lambda\}$, calibrated amplitudes

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, smoothed $\ell_p$ sublevel set, p = 4, $\varepsilon$ = 0.2, $\lambda$ = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.09375, `eta` = 6e-06, batch = 50, 250 iterations, 100 walkers, amplitude scales 1 (constant) and 0.3 (state), tilt 0.

Exact reference: train 0.7496, test 0.7744 (halves 0.7744 / 0.7744), acceptance 0.24, |mean| 1.066, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7226 | 0.7354 | 0.0452 | never | 0.869 | 0.000 | 0 |
| constant `J_a`, `a` = 1 | 0.7130 | 0.7428 | 0.0336 | never | 0.887 | 0.012 | 0 |
| state-dependent `J_g(x)`, `s` = [3.0] | 0.7223 | 0.7360 | 0.0437 | never | 0.873 | 0.000 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.2, $\lambda$ = 4, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.009858, `eta` = 6e-06, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 0.

Exact reference: train 0.7799, test 0.7853 (halves 0.7854 / 0.7853), acceptance 0.26, |mean| 1.737, 1/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7789 | 0.7839 | 0.0023 | 42 | 0.957 | 0.000 | 0 |
| constant `J_a`, `a` = 0.6 | 0.7801 | 0.7849 | 0.0020 | 52 | 0.963 | 0.000 | 0 |
| state-dependent `J_g(x)`, `s` = [1.5, 1.5, 1.5] | 0.7784 | 0.7843 | 0.0016 | 63 | 0.956 | 0.000 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.18, $\lambda$ = 4, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.2104, `eta` = 1e-04, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 3.

Exact reference: train 0.7825, test 0.7610 (halves 0.7610 / 0.7610), acceptance 0.25, |mean| 1.560, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7710 | 0.7446 | 0.0210 | never | 0.329 | 0.000 | 0 |
| constant `J_a`, `a` = 0.6 | 0.7822 | 0.7656 | 0.0135 | 180 | 0.407 | 0.000 | 0 |
| tilted `J_g(x)`, `s` = [0.6, 2.1, 0.6], `c` = 3 | 0.7729 | 0.7504 | 0.0181 | never | 0.437 | 0.000 | 0 |

## smoothed $\ell_p$ sublevel set $\{x : g(x) \leq \lambda\}$, calibrated amplitudes

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, smoothed $\ell_p$ sublevel set, p = 4, $\varepsilon$ = 0.2, $\lambda$ = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.04688, `eta` = 6e-06, batch = 50, 250 iterations, 100 walkers, amplitude scales 1 (constant) and 0.3 (state), tilt 0.

Exact reference: train 0.7483, test 0.7747 (halves 0.7747 / 0.7747), acceptance 0.24, |mean| 1.033, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7156 | 0.7276 | 0.0540 | never | 0.834 | 0.000 | 0 |
| constant `J_a`, `a` = 1 | 0.7126 | 0.7432 | 0.0351 | never | 0.887 | 0.009 | 0 |
| state-dependent `J_g(x)`, `s` = [3.0] | 0.7139 | 0.7297 | 0.0500 | never | 0.829 | 0.000 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.2, $\lambda$ = 4, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.003697, `eta` = 6e-06, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 1.

Exact reference: train 0.7744, test 0.7773 (halves 0.7773 / 0.7773), acceptance 0.24, |mean| 1.528, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7757 | 0.7802 | 0.0032 | 34 | 0.857 | 0.000 | 0 |
| constant `J_a`, `a` = 0.6 | 0.7775 | 0.7823 | 0.0024 | 43 | 0.879 | 0.000 | 0 |
| tilted `J_g(x)`, `s` = [1.5, 1.5, 1.5], `c` = 1 | 0.7749 | 0.7809 | 0.0029 | 66 | 0.924 | 0.000 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.18, $\lambda$ = 4, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.07889, `eta` = 1e-04, batch = 30, 250 iterations, 100 walkers, amplitude scales 0.3 (constant) and 0.3 (state), tilt 3.

Exact reference: train 0.7847, test 0.7579 (halves 0.7577 / 0.7580), acceptance 0.24, |mean| 1.453, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7716 | 0.7453 | 0.0204 | never | 0.319 | 0.000 | 0 |
| constant `J_a`, `a` = 0.6 | 0.7820 | 0.7648 | 0.0146 | 163 | 0.391 | 0.000 | 0 |
| tilted `J_g(x)`, `s` = [0.6, 2.1, 0.6], `c` = 3 | 0.7774 | 0.7513 | 0.0184 | 237 | 0.416 | 0.000 | 0 |

