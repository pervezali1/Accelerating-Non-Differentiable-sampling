# The amplitude each field wants

Selected from the sweeps by the rule in `experiments/select_anchored.py`: the
fewest iterations to the accuracy band, among the amplitudes that do not lose
stationary accuracy and do not fall back on a missed boundary push.  `band` is
iterations to come within 0.005 of the exact constrained lasso posterior's test
accuracy, `error` the whitened distance of the time-averaged mean to its mean,
`error at n/4` the same a quarter of the way in.  `c` is the tilt of the
non-radial `h`; `c = 0` is the paper's own field.  Uncertainties are standard
errors over the seeds.

| problem | set | eta | seeds | field | kappa | c | band | speed-up | error | error at n/4 |
|---|---|---|---|---|---|---|---|---|---|---|
| magic | ball | 6.25e-06 | 3 | `J = 0` | -- | -- | 294 +/- 96 | 1.00x | 1.424 +/- 0.032 | 9.311 |
| magic | ball | 6.25e-06 | 3 | `J_a` | 0.03 | 0 | 294 +/- 96 | **1.00x** | 1.407 +/- 0.033 | 9.079 |
| magic | ball | 6.25e-06 | 3 | `J_s` / `J_g` | 0.03 | 0 | 294 +/- 96 | **1.00x** | 1.410 +/- 0.033 | 9.252 |
| magic | ball | 2.50e-05 | 3 | `J = 0` | -- | -- | 54 +/- 13 | 1.00x | 5.717 +/- 0.302 | 13.113 |
| magic | ball | 2.50e-05 | 3 | `J_a` **(bias traded)** | 0.3 | 0 | 48 +/- 7 | **1.14x** | 10.165 +/- 0.553 | 14.442 |
| magic | ball | 2.50e-05 | 3 | `J_s` / `J_g` **(bias traded)** | 0.3 | 0 | 61 +/- 12 | **0.89x** | 7.966 +/- 0.337 | 14.018 |
| magic | lp | 6.25e-06 | 3 | `J = 0` | -- | -- | 241 +/- 0 | 1.00x | 0.937 +/- 0.032 | 3.583 |
| magic | lp | 6.25e-06 | 3 | `J_a` | 0.03 | 0 | 241 +/- 0 | **1.00x** | 0.919 +/- 0.034 | 3.511 |
| magic | lp | 6.25e-06 | 3 | `J_s` / `J_g` **(bias traded)** | 0.03 | 0 | 241 +/- 0 | **1.00x** | 1.086 +/- 0.024 | 3.535 |
| magic | lp | 2.50e-05 | 3 | `J = 0` | -- | -- | 68 +/- 7 | 1.00x | 2.905 +/- 0.418 | 4.545 |
| magic | lp | 2.50e-05 | 3 | `J_a` **(bias traded)** | 0.3 | 0 | 68 +/- 7 | **1.00x** | 3.538 +/- 0.457 | 4.299 |
| magic | lp | 2.50e-05 | 3 | `J_s` / `J_g` **(bias traded)** | 0.3 | 3 | 214 +/- 18 | **0.32x** | 6.014 +/- 0.229 | 5.853 |
| synthetic | ball | 6.25e-06 | 3 | `J = 0` | -- | -- | 481 +/- 0 | 1.00x | 0.117 +/- 0.013 | 3.415 |
| synthetic | ball | 6.25e-06 | 3 | `J_a` | 1 | 0 | 401 +/- 0 | **1.20x** | 0.074 +/- 0.021 | 2.437 |
| synthetic | ball | 6.25e-06 | 3 | `J_s` / `J_g` | 0.1 | 3 | 428 +/- 27 | **1.12x** | 0.089 +/- 0.017 | 2.919 |
| synthetic | ball | 6.25e-06 | 5 | `J = 0` | -- | -- | 513 +/- 20 | 1.00x | 0.094 +/- 0.018 | 3.490 |
| synthetic | ball | 6.25e-06 | 5 | `J_a` | 1 | 0 | 401 +/- 0 | **1.28x** | 0.057 +/- 0.015 | 2.395 |
| synthetic | ball | 6.25e-06 | 5 | `J_s` / `J_g` | 0.1 | 1 | 401 +/- 0 | **1.28x** | 0.077 +/- 0.015 | 3.289 |
| synthetic | ball | 2.50e-05 | 3 | `J = 0` | -- | -- | 141 +/- 0 | 1.00x | 0.163 +/- 0.019 | 3.398 |
| synthetic | ball | 2.50e-05 | 3 | `J_a` | 1 | 0 | 101 +/- 0 | **1.40x** | 0.112 +/- 0.015 | 2.462 |
| synthetic | ball | 2.50e-05 | 3 | `J_s` / `J_g` | 0.1 | 3 | 134 +/- 13 | **1.05x** | 0.103 +/- 0.010 | 2.904 |
| synthetic | lp | 6.25e-06 | 3 | `J = 0` | -- | -- | 588 +/- 53 | 1.00x | 0.076 +/- 0.038 | 2.482 |
| synthetic | lp | 6.25e-06 | 3 | `J_a` | 1 | 0 | 401 +/- 0 | **1.47x** | 0.077 +/- 0.042 | 1.761 |
| synthetic | lp | 6.25e-06 | 3 | `J_s` / `J_g` | 0.1 | 0 | 561 +/- 0 | **1.05x** | 0.095 +/- 0.038 | 2.376 |
| synthetic | lp | 6.25e-06 | 5 | `J = 0` | -- | -- | 561 +/- 36 | 1.00x | 0.078 +/- 0.025 | 2.501 |
| synthetic | lp | 6.25e-06 | 5 | `J_s` / `J_g` | 0.1 | 2 | 481 +/- 0 | **1.17x** | 0.084 +/- 0.020 | 1.979 |
| synthetic | lp | 2.50e-05 | 3 | `J = 0` | -- | -- | 121 +/- 12 | 1.00x | 0.176 +/- 0.011 | 2.383 |
| synthetic | lp | 2.50e-05 | 3 | `J_a` | 0.3 | 0 | 101 +/- 0 | **1.20x** | 0.175 +/- 0.007 | 2.292 |
| synthetic | lp | 2.50e-05 | 3 | `J_s` / `J_g` | 0.1 | 0 | 128 +/- 7 | **0.95x** | 0.166 +/- 0.040 | 2.277 |
| titanic | ball | 1.00e-04 | 3 | `J = 0` | -- | -- | 388 +/- 68 | 1.00x | 0.518 +/- 0.062 | 11.753 |
| titanic | ball | 1.00e-04 | 3 | `J_a` | 0.6 | 0 | 136 +/- 2 | **2.85x** | 0.538 +/- 0.047 | 5.307 |
| titanic | ball | 1.00e-04 | 3 | `J_s` / `J_g` | 0.3 | 3 | 185 +/- 20 | **2.10x** | 0.495 +/- 0.025 | 8.433 |
| titanic | lp | 1.00e-04 | 5 | `J = 0` | -- | -- | 737 +/- 58 | 1.00x | 0.582 +/- 0.051 | 6.069 |
| titanic | lp | 1.00e-04 | 5 | `J_a` | 1 | 0 | 163 +/- 9 | **4.52x** | 0.418 +/- 0.037 | 2.982 |
| titanic | lp | 1.00e-04 | 5 | `J_s` / `J_g` | 0.3 | 3 | 367 +/- 12 | **2.01x** | 0.441 +/- 0.045 | 3.816 |

Where the rule could not be met:

* magic/ball at `eta` = 2.50e-05: no amplitude of `J_a` improves the band without giving up stationary accuracy; the row shown is the best band.
* magic/ball at `eta` = 2.50e-05: no amplitude of `J_s` / `J_g` improves the band without giving up stationary accuracy; the row shown is the best band.
* magic/lp at `eta` = 6.25e-06: no amplitude of `J_s` / `J_g` improves the band without giving up stationary accuracy; the row shown is the best band.
* magic/lp at `eta` = 2.50e-05: no amplitude of `J_a` improves the band without giving up stationary accuracy; the row shown is the best band.
* magic/lp at `eta` = 2.50e-05: no amplitude of `J_s` / `J_g` improves the band without giving up stationary accuracy; the row shown is the best band.

