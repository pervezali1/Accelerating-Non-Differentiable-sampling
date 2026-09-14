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

`d` = 3, `n_train` = 1600, `n_test` = 400, centred ball, r = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.03125, `eta` = 1e-04, batch = 50, 1000 iterations, 100 walkers, amplitude scale 0.1287.

Exact reference: train 0.7494, test 0.7752 (halves 0.7753 / 0.7752), acceptance 0.24, |mean| 0.973, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7497 | 0.7765 | 0.0047 | 30 | 0.818 | 0.222 | 0 |
| constant `J_a`, `a` = 0.128653 | 0.7498 | 0.7766 | 0.0049 | 29 | 0.819 | 0.222 | 0 |
| state-dependent `J_s(x)`, `s` = [1.2865] | 0.7493 | 0.7753 | 0.0049 | 29 | 0.822 | 0.227 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, centred ball, r = 1.41421, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.003286, `eta` = 1e-04, batch = 30, 1000 iterations, 100 walkers, amplitude scale 0.05725.

Exact reference: train 0.7850, test 0.7910 (halves 0.7910 / 0.7910), acceptance 0.23, |mean| 1.409, 1/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7829 | 0.7906 | 0.0022 | 13 | 0.878 | 0.572 | 0 |
| constant `J_a`, `a` = 0.114501 | 0.7829 | 0.7907 | 0.0020 | 13 | 0.876 | 0.570 | 0 |
| state-dependent `J_s(x)`, `s` = [0.2863, 0.2863, 0.2863] | 0.7831 | 0.7913 | 0.0023 | 13 | 0.879 | 0.584 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, centred ball, r = 1.41421, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.07013, `eta` = 1e-04, batch = 30, 1500 iterations, 100 walkers, amplitude scale 0.8129.

Exact reference: train 0.7808, test 0.7530 (halves 0.7529 / 0.7532), acceptance 0.25, |mean| 1.321, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7797 | 0.7538 | 0.0131 | 486 | 0.311 | 0.036 | 0 |
| constant `J_a`, `a` = 2.4387 | 0.7816 | 0.7631 | 0.0132 | 109 | 0.330 | 0.071 | 77 |
| state-dependent `J_s(x)`, `s` = [1.6258, 5.6903, 1.6258] | 0.7772 | 0.7515 | 0.0171 | 485 | 0.317 | 0.040 | 0 |

## centred ball $K_r = \{x : \|x\|_2^2 \leq r\}$, the paper's own amplitudes

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, centred ball, r = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.03125, `eta` = 1e-04, batch = 50, 1000 iterations, 100 walkers, amplitude scale 1.

Exact reference: train 0.7494, test 0.7752 (halves 0.7753 / 0.7752), acceptance 0.24, |mean| 0.973, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7497 | 0.7765 | 0.0047 | 30 | 0.818 | 0.222 | 0 |
| constant `J_a`, `a` = 1 | 0.7490 | 0.7763 | 0.0056 | 20 | 0.820 | 0.209 | 0 |
| state-dependent `J_s(x)`, `s` = [10.0] | 0.6641 | 0.6365 | 0.0426 | never | 0.898 | 0.977 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, centred ball, r = 1.41421, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.003286, `eta` = 1e-04, batch = 30, 1000 iterations, 100 walkers, amplitude scale 1.

Exact reference: train 0.7850, test 0.7910 (halves 0.7910 / 0.7910), acceptance 0.23, |mean| 1.409, 1/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7829 | 0.7906 | 0.0022 | 13 | 0.878 | 0.572 | 0 |
| constant `J_a`, `a` = 2 | 0.7376 | 0.7426 | 0.0039 | never | 0.932 | 0.778 | 72303 |
| state-dependent `J_s(x)`, `s` = [5.0, 5.0, 5.0] | 0.5348 | 0.5266 | 0.0171 | never | 0.893 | 0.981 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, centred ball, r = 1.41421, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.07013, `eta` = 1e-04, batch = 30, 1500 iterations, 100 walkers, amplitude scale 1.

Exact reference: train 0.7808, test 0.7530 (halves 0.7529 / 0.7532), acceptance 0.25, |mean| 1.321, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7797 | 0.7538 | 0.0131 | 486 | 0.311 | 0.036 | 0 |
| constant `J_a`, `a` = 3 | 0.7804 | 0.7607 | 0.0158 | 97 | 0.326 | 0.066 | 297 |
| state-dependent `J_s(x)`, `s` = [2.0, 7.0, 2.0] | 0.7767 | 0.7494 | 0.0171 | 424 | 0.319 | 0.042 | 0 |

## smoothed $\ell_p$ sublevel set $\{x : g(x) \leq \lambda\}$, calibrated amplitudes

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, smoothed $\ell_p$ sublevel set, p = 4, $\varepsilon$ = 0.2, $\lambda$ = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.03125, `eta` = 1e-04, batch = 50, 1000 iterations, 100 walkers, amplitude scale 0.134.

Exact reference: train 0.7497, test 0.7755 (halves 0.7755 / 0.7755), acceptance 0.24, |mean| 1.053, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7472 | 0.7738 | 0.0045 | 30 | 0.839 | 0.082 | 0 |
| constant `J_a`, `a` = 0.13398 | 0.7472 | 0.7739 | 0.0047 | 29 | 0.839 | 0.082 | 0 |
| state-dependent `J_g(x)`, `s` = [1.3398] | 0.7388 | 0.7628 | 0.0152 | 31 | 0.888 | 0.190 | 0 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.2, $\lambda$ = 4, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.003286, `eta` = 1e-04, batch = 30, 1000 iterations, 100 walkers, amplitude scale 0.03783.

Exact reference: train 0.7837, test 0.7912 (halves 0.7912 / 0.7912), acceptance 0.23, |mean| 1.751, 4/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7835 | 0.7880 | 0.0012 | 16 | 0.891 | 0.157 | 0 |
| constant `J_a`, `a` = 0.0756555 | 0.7835 | 0.7874 | 0.0012 | 16 | 0.890 | 0.154 | 0 |
| state-dependent `J_g(x)`, `s` = [0.1891, 0.1891, 0.1891] | 0.7823 | 0.7878 | 0.0015 | 16 | 0.899 | 0.183 | 0 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.18, $\lambda$ = 4, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.07013, `eta` = 1e-04, batch = 30, 2000 iterations, 100 walkers, amplitude scale 0.7555.

Exact reference: train 0.7819, test 0.7566 (halves 0.7564 / 0.7567), acceptance 0.23, |mean| 1.471, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7815 | 0.7543 | 0.0145 | 839 | 0.327 | 0.000 | 0 |
| constant `J_a`, `a` = 1.51105 | 0.7820 | 0.7576 | 0.0159 | 179 | 0.382 | 0.009 | 0 |
| state-dependent `J_g(x)`, `s` = [1.511, 5.2887, 1.511] | 0.7830 | 0.7615 | 0.0137 | 462 | 0.343 | 0.000 | 0 |

## smoothed $\ell_p$ sublevel set $\{x : g(x) \leq \lambda\}$, the paper's own amplitudes

### Synthetic ($d$ = 3)

`d` = 3, `n_train` = 1600, `n_test` = 400, smoothed $\ell_p$ sublevel set, p = 4, $\varepsilon$ = 0.2, $\lambda$ = 1, `lam` = 16 (= 0.01 `n`), `delta` = 0.03125, `eta` = 1e-04, batch = 50, 1000 iterations, 100 walkers, amplitude scale 1.

Exact reference: train 0.7497, test 0.7755 (halves 0.7755 / 0.7755), acceptance 0.24, |mean| 1.053, 0/3 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7472 | 0.7738 | 0.0045 | 30 | 0.839 | 0.082 | 0 |
| constant `J_a`, `a` = 1 | 0.7466 | 0.7729 | 0.0045 | 22 | 0.836 | 0.097 | 0 |
| state-dependent `J_g(x)`, `s` = [10.0] | 0.5112 | 0.4968 | 0.0875 | never | 0.930 | 0.985 | 9239 |

### MAGIC Gamma Telescope

`d` = 9, `n_train` = 15216, `n_test` = 3804, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.2, $\lambda$ = 4, `lam` = 152.2 (= 0.01 `n`), `delta` = 0.003286, `eta` = 1e-04, batch = 30, 1000 iterations, 100 walkers, amplitude scale 1.

Exact reference: train 0.7837, test 0.7912 (halves 0.7912 / 0.7912), acceptance 0.23, |mean| 1.751, 4/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7835 | 0.7880 | 0.0012 | 16 | 0.891 | 0.157 | 0 |
| constant `J_a`, `a` = 2 | 0.6752 | 0.6782 | 0.0138 | never | 0.939 | 0.684 | 58415 |
| state-dependent `J_g(x)`, `s` = [5.0, 5.0, 5.0] | 0.5397 | 0.5391 | 0.0697 | never | 0.912 | 0.998 | 32099 |

### Titanic (survival)

`d` = 9, `n_train` = 713, `n_test` = 178, smoothed $\ell_p$ sublevel set, p = 2.4, $\varepsilon$ = 0.18, $\lambda$ = 4, `lam` = 7.13 (= 0.01 `n`), `delta` = 0.07013, `eta` = 1e-04, batch = 30, 2000 iterations, 100 walkers, amplitude scale 1.

Exact reference: train 0.7819, test 0.7566 (halves 0.7564 / 0.7567), acceptance 0.23, |mean| 1.471, 0/9 coordinates within 0.01 of zero.

| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |
|---|---|---|---|---|---|---|---|
| `J = 0` (anchored PSGLD) | 0.7815 | 0.7543 | 0.0145 | 839 | 0.327 | 0.000 | 0 |
| constant `J_a`, `a` = 2 | 0.7829 | 0.7629 | 0.0143 | 168 | 0.378 | 0.011 | 2 |
| state-dependent `J_g(x)`, `s` = [2.0, 7.0, 2.0] | 0.7785 | 0.7570 | 0.0174 | 422 | 0.354 | 0.003 | 0 |

