# The amplitude each field wants

Selected from the sweeps by the rule in `experiments/select_anchored.py`: the
fewest iterations to the accuracy band, among the amplitudes that do not lose
stationary accuracy and do not fall back on a missed boundary push.  `band` is
iterations to come within 0.005 of the exact constrained lasso posterior's test
accuracy, `error` the whitened distance of the time-averaged mean to its mean,
`error at n/4` the same a quarter of the way in.  `c` is the tilt of the
non-radial `h`; `c = 0` is the paper's own field.  Uncertainties are standard
errors over the seeds.

| problem | set | penalty | eta | seeds | field | kappa | c | band | speed-up | gap at the end | gap ratio | error | error ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| magic | ball | group | 6.25e-06 | 3 | `J = 0` | -- | -- | 109 +/- 6 | 1.00x | 0.0018 +/- 0.0011 | 1.00x | 36.090 +/- 2.540 | 1.00x |
| magic | ball | group | 6.25e-06 | 3 | `J_a` | 0.3 | 0 | 102 +/- 4 | **1.07x** | 0.0010 +/- 0.0008 | **1.72x** | 15.412 +/- 1.296 | **2.34x** |
| magic | ball | group | 6.25e-06 | 3 | `J_s` / `J_g` | 0.3 | 0 | 102 +/- 11 | **1.07x** | 0.0007 +/- 0.0004 | **2.42x** | 32.102 +/- 2.785 | **1.12x** |
| magic | lp | group | 6.25e-06 | 3 | `J = 0` | -- | -- | 104 +/- 15 | 1.00x | 0.0014 +/- 0.0005 | 1.00x | 11.766 +/- 0.342 | 1.00x |
| magic | lp | group | 6.25e-06 | 3 | `J_a` | 0.3 | 0 | 116 +/- 8 | **0.89x** | 0.0006 +/- 0.0001 | **2.21x** | 11.285 +/- 0.767 | **1.04x** |
| magic | lp | group | 6.25e-06 | 3 | `J_s` / `J_g` | 0.3 | 0 | 128 +/- 16 | **0.81x** | 0.0031 +/- 0.0014 | **0.47x** | 11.132 +/- 0.545 | **1.06x** |
| synthetic | ball | group | 6.25e-06 | 3 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0529 +/- 0.0011 | 1.00x | 25.556 +/- 0.240 | 1.00x |
| synthetic | ball | group | 6.25e-06 | 3 | `J_a` | 1 | 0 | 250 +/- 0 | **1.00x** | 0.0358 +/- 0.0014 | **1.48x** | 21.789 +/- 0.278 | **1.17x** |
| synthetic | ball | group | 6.25e-06 | 3 | `J_s` / `J_g` | 0.3 | 1 | 250 +/- 0 | **1.00x** | 0.0518 +/- 0.0012 | **1.02x** | 24.378 +/- 0.235 | **1.05x** |
| synthetic | lp | group | 6.25e-06 | 3 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0551 +/- 0.0053 | 1.00x | 15.477 +/- 0.341 | 1.00x |
| synthetic | lp | group | 6.25e-06 | 3 | `J_a` | 1 | 0 | 250 +/- 0 | **1.00x** | 0.0418 +/- 0.0025 | **1.32x** | 15.090 +/- 0.295 | **1.03x** |
| synthetic | lp | group | 6.25e-06 | 3 | `J_s` / `J_g` | 0.3 | 0 | 250 +/- 0 | **1.00x** | 0.0541 +/- 0.0053 | **1.02x** | 15.350 +/- 0.389 | **1.01x** |
| titanic | ball | group | 1.00e-04 | 5 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0160 +/- 0.0023 | 1.00x | 11.650 +/- 0.277 | 1.00x |
| titanic | ball | group | 1.00e-04 | 5 | `J_a` | 0.3 | 0 | 154 +/- 10 | **1.62x** | 0.0029 +/- 0.0008 | **5.52x** | 5.177 +/- 0.038 | **2.25x** |
| titanic | ball | group | 1.00e-04 | 5 | `J_s` / `J_g` | 0.3 | 3 | 198 +/- 17 | **1.26x** | 0.0049 +/- 0.0010 | **3.25x** | 6.563 +/- 0.108 | **1.78x** |
| titanic | lp | group | 1.00e-04 | 5 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0236 +/- 0.0020 | 1.00x | 6.460 +/- 0.073 | 1.00x |
| titanic | lp | group | 1.00e-04 | 5 | `J_a` | 0.3 | 0 | 188 +/- 13 | **1.33x** | 0.0034 +/- 0.0011 | **6.94x** | 5.382 +/- 0.119 | **1.20x** |
| titanic | lp | group | 1.00e-04 | 5 | `J_s` / `J_g` | 0.3 | 3 | 250 +/- 0 | **1.00x** | 0.0171 +/- 0.0023 | **1.38x** | 4.034 +/- 0.131 | **1.60x** |

