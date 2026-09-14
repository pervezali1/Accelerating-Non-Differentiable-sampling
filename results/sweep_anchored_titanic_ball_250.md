# Amplitude sweep: titanic, ball

`lam` = 7.13, `delta` = 0.07013, 100 walkers, batch 30, seeds [0, 1, 2, 3, 4], profile `paper`, block order `columns`.  `kappa` multiplies the paper's own amplitudes (`a` = 3, `s` = [2.0, 7.0, 2.0]).

| eta | iterations | field | kappa | tilt | band | gap at the tail | error | running error | accuracy | boundary | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.00e-04 | 250 | `zero` | 0 | 0 | 250 +/- 0 (not all) | 0.0167 +/- 0.0022 | 10.931 +/- 0.207 | 14.589 | 0.7363 | 0.000 | 0 |
| 1.00e-04 | 250 | `constant` | 0.3 | 0 | 156 +/- 11 | 0.0032 +/- 0.0008 | 5.948 +/- 0.066 | 10.613 | 0.7535 | 0.084 | 0 |
| 1.00e-04 | 250 | `constant` | 1 | 0 | 98 +/- 2 | 0.0049 +/- 0.0010 | 2.639 +/- 0.087 | 5.823 | 0.7548 | 0.112 | 269 |
| 1.00e-04 | 250 | `constant` | 2 | 0 | 54 +/- 3 | 0.0092 +/- 0.0028 | 3.170 +/- 0.164 | 4.841 | 0.7461 | 0.131 | 2233 |
| 1.00e-04 | 250 | `constant` | 3 | 0 | 36 +/- 3 | 0.0054 +/- 0.0019 | 3.686 +/- 0.196 | 4.761 | 0.7522 | 0.206 | 4690 |
