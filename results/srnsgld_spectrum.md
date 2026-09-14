# Rate redistribution: what each field can actually buy

Eigenvalues of the linearised drift ``(I + J(x*)) H`` at the constrained MAP.
`rate` is `min Re mu`, the exponential rate of the slowest direction; `gain` is
that rate over `lambda_min`, the reversible one; `radius` is `max |mu|`, which is
what limits the step size.  `measured` is the ratio of the `J = 0` whitened error
to this field's, a tenth of the way into the runs of `run_srnsgld.py` -- the
column `gain` is what it is meant to predict.

## Titanic (survival)

`lambda_min` = 13.7, `lambda_max` = 190.1, condition number 13.9, `|x*|` = 1.734 of `r` = 2.

| field | rho | null directions of `J` | rate | gain | radius | measured at 200 |
|---|---|---|---|---|---|---|
| `J = 0` | 0 | 9 | 13.7 | 1.00 | 190 | 1.00 |
| `constant` | 0.25 | 1 | 14.1 | 1.03 | 182 | 1.15 |
| `constant` | 0.5 | 1 | 15.5 | 1.13 | 163 | 1.38 |
| `constant` | 1 | 1 | 19.6 | 1.44 | 159 | 1.91 |
| `constant` | 2 | 1 | 28.7 | 2.10 | 238 | 2.04 |
| `constant` | 4 | 1 | 39.1 | 2.86 | 432 | 0.33 |
| `state` | 0.25 | 3 | 13.7 | 1.01 | 183 | 1.00 |
| `state` | 0.5 | 3 | 14.0 | 1.02 | 168 | 1.00 |
| `state` | 1 | 3 | 14.7 | 1.07 | 158 | 1.04 |
| `state` | 2 | 3 | 16.4 | 1.20 | 209 | 1.14 |
| `state` | 4 | 3 | 19.0 | 1.39 | 384 | 0.34 |

## MAGIC Gamma Telescope

`lambda_min` = 192.0, `lambda_max` = 6553.8, condition number 34.1, `|x*|` = 2.000 of `r` = 2.

| field | rho | null directions of `J` | rate | gain | radius | measured at 200 |
|---|---|---|---|---|---|---|
| `J = 0` | 0 | 9 | 192.0 | 1.00 | 6554 | 1.00 |
| `constant` | 0.25 | 1 | 206.6 | 1.08 | 6527 | 1.06 |
| `constant` | 0.5 | 1 | 261.9 | 1.36 | 6445 | 1.15 |
| `constant` | 1 | 1 | 304.5 | 1.59 | 6093 | 1.44 |
| `constant` | 2 | 1 | 388.6 | 2.02 | 3522 | 1.97 |
| `constant` | 4 | 1 | 488.7 | 2.55 | 6682 | 2.41 |
| `state` | 0.25 | 3 | 218.4 | 1.14 | 6543 | 1.00 |
| `state` | 0.5 | 3 | 266.5 | 1.39 | 6511 | 1.01 |
| `state` | 1 | 3 | 300.3 | 1.56 | 6379 | 1.02 |
| `state` | 2 | 3 | 325.3 | 1.69 | 5814 | 1.06 |
| `state` | 4 | 3 | 329.6 | 1.72 | 4425 | 1.16 |

