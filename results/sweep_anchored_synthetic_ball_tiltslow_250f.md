# Amplitude sweep: synthetic, ball

`lam` = 16, `delta` = 0.03125, 100 walkers, batch 50, seeds [0, 1, 2, 3, 4], profile `paper`, block order `columns`.  `kappa` multiplies the paper's own amplitudes (`a` = 1, `s` = [10.0]).

| eta | iterations | field | kappa | tilt | band | gap at the tail | error | running error | accuracy | boundary | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6.25e-06 | 250 | `zero` | 0 | 0 | 250 +/- 0 (not all) | 0.0653 +/- 0.0033 | 24.436 +/- 0.347 | 30.426 | 0.7099 | 0.000 | 0 |
| 6.25e-06 | 250 | `state` | 0.01 | 1 | 250 +/- 0 (not all) | 0.0657 +/- 0.0031 | 24.436 +/- 0.334 | 30.427 | 0.7096 | 0.000 | 0 |
| 6.25e-06 | 250 | `state` | 0.01 | 3 | 250 +/- 0 (not all) | 0.0663 +/- 0.0031 | 24.454 +/- 0.321 | 30.438 | 0.7089 | 0.000 | 0 |
| 6.25e-06 | 250 | `state` | 0.03 | 1 | 250 +/- 0 (not all) | 0.0661 +/- 0.0023 | 24.461 +/- 0.309 | 30.443 | 0.7092 | 0.000 | 0 |
| 6.25e-06 | 250 | `state` | 0.03 | 3 | 250 +/- 0 (not all) | 0.0674 +/- 0.0025 | 24.517 +/- 0.286 | 30.475 | 0.7079 | 0.000 | 0 |
