# The amplitude each field wants

Selected from the sweeps by the rule in `experiments/select_anchored.py`: the
fewest iterations to the accuracy band, among the amplitudes that do not lose
stationary accuracy and do not fall back on a missed boundary push.  `band` is
iterations to come within 0.005 of the exact constrained lasso posterior's test
accuracy, `error` the whitened distance of the time-averaged mean to its mean,
`error at n/4` the same a quarter of the way in.  `c` is the tilt of the
non-radial `h`; `c = 0` is the paper's own field.  Uncertainties are standard
errors over the seeds.

| problem | set | eta | seeds | field | kappa | c | band | speed-up | gap at the end | gap ratio | error | error ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| magic | ball | 6.25e-06 | 3 | `J = 0` | -- | -- | 167 +/- 24 | 1.00x | 0.0047 +/- 0.0010 | 1.00x | 59.254 +/- 2.426 | 1.00x |
| magic | ball | 6.25e-06 | 3 | `J_a` | 0.3 | 0 | 143 +/- 9 | **1.17x** | 0.0031 +/- 0.0007 | **1.51x** | 19.152 +/- 1.559 | **3.09x** |
| magic | ball | 6.25e-06 | 3 | `J_s` / `J_g` | 0.3 | 0 | 147 +/- 9 | **1.13x** | 0.0042 +/- 0.0003 | **1.12x** | 50.697 +/- 2.564 | **1.17x** |
| magic | ball | 2.50e-05 | 3 | `J = 0` | -- | -- | 54 +/- 13 | 1.00x | nan +/- nan | 1.00x | 5.717 +/- 0.302 | 1.00x |
| magic | ball | 2.50e-05 | 3 | `J_a` **(bias traded)** | 0.3 | 0 | 48 +/- 7 | **1.14x** | nan +/- nan | **nanx** | 10.165 +/- 0.553 | **0.56x** |
| magic | ball | 2.50e-05 | 3 | `J_s` / `J_g` **(bias traded)** | 0.3 | 0 | 61 +/- 12 | **0.89x** | nan +/- nan | **nanx** | 7.966 +/- 0.337 | **0.72x** |
| magic | lp | 6.25e-06 | 3 | `J = 0` | -- | -- | 178 +/- 7 | 1.00x | 0.0042 +/- 0.0004 | 1.00x | 17.049 +/- 0.133 | 1.00x |
| magic | lp | 6.25e-06 | 3 | `J_a` | 0.3 | 0 | 165 +/- 16 | **1.08x** | 0.0033 +/- 0.0010 | **1.30x** | 15.301 +/- 0.839 | **1.11x** |
| magic | lp | 6.25e-06 | 3 | `J_s` / `J_g` | 0.3 | 0 | 164 +/- 8 | **1.09x** | 0.0063 +/- 0.0009 | **0.67x** | 16.292 +/- 0.287 | **1.05x** |
| magic | lp | 2.50e-05 | 3 | `J = 0` | -- | -- | 68 +/- 7 | 1.00x | nan +/- nan | 1.00x | 2.905 +/- 0.418 | 1.00x |
| magic | lp | 2.50e-05 | 3 | `J_a` **(bias traded)** | 0.3 | 0 | 68 +/- 7 | **1.00x** | nan +/- nan | **nanx** | 3.538 +/- 0.457 | **0.82x** |
| magic | lp | 2.50e-05 | 3 | `J_s` / `J_g` **(bias traded)** | 0.3 | 3 | 214 +/- 18 | **0.32x** | nan +/- nan | **nanx** | 6.014 +/- 0.229 | **0.48x** |
| synthetic | ball | 6.25e-06 | 3 | `J = 0` | -- | -- | 481 +/- 0 | 1.00x | nan +/- nan | 1.00x | 0.117 +/- 0.013 | 1.00x |
| synthetic | ball | 6.25e-06 | 3 | `J_a` | 1 | 0 | 401 +/- 0 | **1.20x** | nan +/- nan | **nanx** | 0.074 +/- 0.021 | **1.59x** |
| synthetic | ball | 6.25e-06 | 3 | `J_s` / `J_g` | 0.1 | 3 | 428 +/- 27 | **1.12x** | nan +/- nan | **nanx** | 0.089 +/- 0.017 | **1.32x** |
| synthetic | ball | 6.25e-06 | 5 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0653 +/- 0.0033 | 1.00x | 24.436 +/- 0.347 | 1.00x |
| synthetic | ball | 6.25e-06 | 5 | `J_a` | 1 | 0 | 401 +/- 0 | **0.62x** | nan +/- nan | **65345337.50x** | 0.057 +/- 0.015 | **430.16x** |
| synthetic | ball | 6.25e-06 | 5 | `J_s` / `J_g` | 0.1 | 1 | 401 +/- 0 | **0.62x** | nan +/- nan | **65345337.50x** | 0.077 +/- 0.015 | **317.46x** |
| synthetic | ball | 2.50e-05 | 3 | `J = 0` | -- | -- | 141 +/- 0 | 1.00x | nan +/- nan | 1.00x | 0.163 +/- 0.019 | 1.00x |
| synthetic | ball | 2.50e-05 | 3 | `J_a` | 1 | 0 | 101 +/- 0 | **1.40x** | nan +/- nan | **nanx** | 0.112 +/- 0.015 | **1.45x** |
| synthetic | ball | 2.50e-05 | 3 | `J_s` / `J_g` | 0.1 | 3 | 134 +/- 13 | **1.05x** | nan +/- nan | **nanx** | 0.103 +/- 0.010 | **1.58x** |
| synthetic | ball | 2.50e-05 | 5 | `J = 0` | -- | -- | 125 +/- 3 | 1.00x | 0.0010 +/- 0.0005 | 1.00x | 5.058 +/- 0.110 | 1.00x |
| synthetic | ball | 2.50e-05 | 5 | `J_a` | 1 | 0 | 89 +/- 2 | **1.40x** | 0.0024 +/- 0.0005 | **0.42x** | 1.907 +/- 0.100 | **2.65x** |
| synthetic | ball | 2.50e-05 | 5 | `J_s` / `J_g` | 0.3 | 0 | 125 +/- 8 | **1.00x** | 0.0040 +/- 0.0012 | **0.25x** | 5.024 +/- 0.076 | **1.01x** |
| synthetic | lp | 6.25e-06 | 3 | `J = 0` | -- | -- | 588 +/- 53 | 1.00x | nan +/- nan | 1.00x | 0.076 +/- 0.038 | 1.00x |
| synthetic | lp | 6.25e-06 | 3 | `J_a` | 1 | 0 | 401 +/- 0 | **1.47x** | nan +/- nan | **nanx** | 0.077 +/- 0.042 | **0.99x** |
| synthetic | lp | 6.25e-06 | 3 | `J_s` / `J_g` | 0.1 | 0 | 561 +/- 0 | **1.05x** | nan +/- nan | **nanx** | 0.095 +/- 0.038 | **0.80x** |
| synthetic | lp | 6.25e-06 | 5 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0623 +/- 0.0031 | 1.00x | 14.473 +/- 0.183 | 1.00x |
| synthetic | lp | 6.25e-06 | 5 | `J_a` | 3 | 0 | 200 +/- 9 | **1.25x** | 0.0030 +/- 0.0008 | **20.92x** | 9.614 +/- 0.239 | **1.51x** |
| synthetic | lp | 6.25e-06 | 5 | `J_s` / `J_g` | 0.1 | 2 | 481 +/- 0 | **0.52x** | nan +/- nan | **62328425.00x** | 0.084 +/- 0.020 | **172.60x** |
| synthetic | lp | 2.50e-05 | 3 | `J = 0` | -- | -- | 121 +/- 12 | 1.00x | nan +/- nan | 1.00x | 0.176 +/- 0.011 | 1.00x |
| synthetic | lp | 2.50e-05 | 3 | `J_a` | 0.3 | 0 | 101 +/- 0 | **1.20x** | nan +/- nan | **nanx** | 0.175 +/- 0.007 | **1.01x** |
| synthetic | lp | 2.50e-05 | 3 | `J_s` / `J_g` | 0.1 | 0 | 128 +/- 7 | **0.95x** | nan +/- nan | **nanx** | 0.166 +/- 0.040 | **1.06x** |
| synthetic | lp | 2.50e-05 | 5 | `J = 0` | -- | -- | 120 +/- 7 | 1.00x | 0.0024 +/- 0.0006 | 1.00x | 4.078 +/- 0.232 | 1.00x |
| synthetic | lp | 2.50e-05 | 5 | `J_a` | 1 | 0 | 89 +/- 2 | **1.35x** | 0.0037 +/- 0.0011 | **0.63x** | 2.118 +/- 0.164 | **1.93x** |
| synthetic | lp | 2.50e-05 | 5 | `J_s` / `J_g` | 0.3 | 0 | 128 +/- 5 | **0.93x** | 0.0120 +/- 0.0045 | **0.20x** | 3.790 +/- 0.257 | **1.08x** |
| titanic | ball | 1.00e-04 | 3 | `J = 0` | -- | -- | 388 +/- 68 | 1.00x | nan +/- nan | 1.00x | 0.518 +/- 0.062 | 1.00x |
| titanic | ball | 1.00e-04 | 3 | `J_a` | 0.6 | 0 | 136 +/- 2 | **2.85x** | nan +/- nan | **nanx** | 0.538 +/- 0.047 | **0.96x** |
| titanic | ball | 1.00e-04 | 3 | `J_s` / `J_g` | 0.3 | 3 | 185 +/- 20 | **2.10x** | nan +/- nan | **nanx** | 0.495 +/- 0.025 | **1.05x** |
| titanic | ball | 1.00e-04 | 5 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0167 +/- 0.0022 | 1.00x | 10.931 +/- 0.207 | 1.00x |
| titanic | ball | 1.00e-04 | 5 | `J_a` | 0.3 | 0 | 156 +/- 11 | **1.60x** | 0.0032 +/- 0.0008 | **5.26x** | 5.948 +/- 0.066 | **1.84x** |
| titanic | ball | 1.00e-04 | 5 | `J_s` / `J_g` | 0.3 | 3 | 191 +/- 12 | **1.31x** | 0.0038 +/- 0.0013 | **4.43x** | 6.658 +/- 0.109 | **1.64x** |
| titanic | lp | 1.00e-04 | 5 | `J = 0` | -- | -- | 737 +/- 58 | 1.00x | nan +/- nan | 1.00x | 0.582 +/- 0.051 | 1.00x |
| titanic | lp | 1.00e-04 | 5 | `J_a` | 1 | 0 | 163 +/- 9 | **4.52x** | nan +/- nan | **nanx** | 0.418 +/- 0.037 | **1.39x** |
| titanic | lp | 1.00e-04 | 5 | `J_s` / `J_g` | 0.3 | 3 | 367 +/- 12 | **2.01x** | nan +/- nan | **nanx** | 0.441 +/- 0.045 | **1.32x** |

Where the rule could not be met:

* magic/ball at `eta` = 2.50e-05: no amplitude of `J_a` improves the band without giving up stationary accuracy; the row shown is the best band.
* magic/ball at `eta` = 2.50e-05: no amplitude of `J_s` / `J_g` improves the band without giving up stationary accuracy; the row shown is the best band.
* magic/lp at `eta` = 2.50e-05: no amplitude of `J_a` improves the band without giving up stationary accuracy; the row shown is the best band.
* magic/lp at `eta` = 2.50e-05: no amplitude of `J_s` / `J_g` improves the band without giving up stationary accuracy; the row shown is the best band.

