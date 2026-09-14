# Constrained logistic regression in d = 9: PSGLD against SRNSGLD

One step size per dataset from the curvature at the constrained MAP, shared by
all three fields; one mini-batch stream per iteration, shared by all walkers;
walkers started uniform on the centred unit ball.  `error` is the whitened
distance from the running posterior mean to the exact constrained posterior's.

## Titanic (survival)

`d` = 9, `n_train` = 623, `n_test` = 268, `r` = 2.0, `h` = 1.578e-03, batch = 256, 2000 iterations, 32 walkers.

Exact constrained posterior: accuracy 0.7500 (halves 0.7500 / 0.7500), loss 0.5002, acceptance 0.22, |mean| 1.740 of `r` = 2.0.

| field | rho | a | accuracy | loss | error | boundary hits | missed reflections |
|---|---|---|---|---|---|---|---|
| zero | 0 | 0.000 | 0.7500 | 0.5005 | 0.1400 | 0.057 | 0 |
| constant | 0.25 | 0.131 | 0.7500 | 0.5005 | 0.1415 | 0.058 | 0 |
| constant | 0.5 | 0.263 | 0.7500 | 0.5005 | 0.1406 | 0.059 | 0 |
| constant | 1 | 0.526 | 0.7500 | 0.5004 | 0.1347 | 0.065 | 0 |
| constant | 2 | 1.051 | 0.7463 | 0.4999 | 0.1487 | 0.089 | 4 |
| constant | 4 | 2.103 | 0.7388 | 0.4968 | 1.4670 | 0.233 | 9424 |
| state | 0.25 | 0.167 | 0.7500 | 0.5004 | 0.1389 | 0.057 | 0 |
| state | 0.5 | 0.335 | 0.7500 | 0.5004 | 0.1379 | 0.058 | 0 |
| state | 1 | 0.669 | 0.7500 | 0.5003 | 0.1397 | 0.061 | 0 |
| state | 2 | 1.338 | 0.7500 | 0.5001 | 0.1856 | 0.077 | 0 |
| state | 4 | 2.676 | 0.7425 | 0.5026 | 1.3708 | 0.562 | 0 |

## MAGIC Gamma Telescope

`d` = 9, `n_train` = 13314, `n_test` = 5706, `r` = 2.0, `h` = 4.577e-05, batch = 4096, 2000 iterations, 32 walkers.

Exact constrained posterior: accuracy 0.7860 (halves 0.7860 / 0.7860), loss 0.4874, acceptance 0.23, |mean| 1.987 of `r` = 2.0.

| field | rho | a | accuracy | loss | error | boundary hits | missed reflections |
|---|---|---|---|---|---|---|---|
| zero | 0 | 0.000 | 0.7862 | 0.4870 | 1.9767 | 0.452 | 0 |
| constant | 0.25 | 0.131 | 0.7862 | 0.4871 | 1.8585 | 0.455 | 0 |
| constant | 0.5 | 0.263 | 0.7860 | 0.4871 | 1.6844 | 0.458 | 0 |
| constant | 1 | 0.526 | 0.7864 | 0.4872 | 1.3242 | 0.462 | 0 |
| constant | 2 | 1.051 | 0.7860 | 0.4873 | 1.0160 | 0.450 | 0 |
| constant | 4 | 2.103 | 0.7860 | 0.4873 | 1.4386 | 0.372 | 50 |
| state | 0.25 | 0.167 | 0.7862 | 0.4870 | 1.9697 | 0.453 | 0 |
| state | 0.5 | 0.335 | 0.7862 | 0.4870 | 1.9554 | 0.453 | 0 |
| state | 1 | 0.669 | 0.7858 | 0.4870 | 1.9182 | 0.456 | 0 |
| state | 2 | 1.338 | 0.7858 | 0.4870 | 1.8530 | 0.459 | 0 |
| state | 4 | 2.676 | 0.7858 | 0.4870 | 1.7550 | 0.471 | 0 |

