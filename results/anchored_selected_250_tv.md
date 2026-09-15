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
| magic | ball | tv | 6.25e-06 | 3 | `J = 0` | -- | -- | 82 +/- 6 | 1.00x | 0.0023 +/- 0.0006 | 1.00x | 30.432 +/- 2.000 | 1.00x |
| magic | ball | tv | 6.25e-06 | 3 | `J_a` | 0.3 | 0 | 88 +/- 5 | **0.93x** | 0.0030 +/- 0.0015 | **0.79x** | 17.516 +/- 1.943 | **1.74x** |
| magic | ball | tv | 6.25e-06 | 3 | `J_s` / `J_g` | 0.3 | 0 | 76 +/- 1 | **1.08x** | 0.0023 +/- 0.0013 | **1.04x** | 29.201 +/- 2.451 | **1.04x** |
| magic | lp | tv | 6.25e-06 | 3 | `J = 0` | -- | -- | 84 +/- 17 | 1.00x | 0.0019 +/- 0.0010 | 1.00x | 12.898 +/- 0.718 | 1.00x |
| magic | lp | tv | 6.25e-06 | 3 | `J_a` | 0.3 | 0 | 88 +/- 2 | **0.95x** | 0.0049 +/- 0.0010 | **0.40x** | 13.294 +/- 0.658 | **0.97x** |
| magic | lp | tv | 6.25e-06 | 3 | `J_s` / `J_g` | 0.3 | 1 | 122 +/- 21 | **0.69x** | 0.0025 +/- 0.0015 | **0.78x** | 9.948 +/- 1.544 | **1.30x** |
| synthetic | ball | tv | 6.25e-06 | 3 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0636 +/- 0.0018 | 1.00x | 20.961 +/- 0.248 | 1.00x |
| synthetic | ball | tv | 6.25e-06 | 3 | `J_a` | 1 | 0 | 250 +/- 0 | **1.00x** | 0.0394 +/- 0.0018 | **1.61x** | 18.473 +/- 0.262 | **1.13x** |
| synthetic | ball | tv | 6.25e-06 | 3 | `J_s` / `J_g` | 0.3 | 1 | 250 +/- 0 | **1.00x** | 0.0602 +/- 0.0016 | **1.06x** | 20.096 +/- 0.178 | **1.04x** |
| synthetic | lp | tv | 6.25e-06 | 3 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0650 +/- 0.0065 | 1.00x | 13.078 +/- 0.329 | 1.00x |
| synthetic | lp | tv | 6.25e-06 | 3 | `J_a` | 1 | 0 | 250 +/- 0 | **1.00x** | 0.0434 +/- 0.0030 | **1.50x** | 13.201 +/- 0.268 | **0.99x** |
| synthetic | lp | tv | 6.25e-06 | 3 | `J_s` / `J_g` | 0.3 | 0 | 250 +/- 0 | **1.00x** | 0.0647 +/- 0.0057 | **1.00x** | 13.127 +/- 0.310 | **1.00x** |
| titanic | ball | tv | 1.00e-04 | 5 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0156 +/- 0.0020 | 1.00x | 9.489 +/- 0.203 | 1.00x |
| titanic | ball | tv | 1.00e-04 | 5 | `J_a` | 0.3 | 0 | 142 +/- 9 | **1.76x** | 0.0031 +/- 0.0012 | **5.10x** | 5.369 +/- 0.060 | **1.77x** |
| titanic | ball | tv | 1.00e-04 | 5 | `J_s` / `J_g` | 0.3 | 1 | 209 +/- 12 | **1.20x** | 0.0045 +/- 0.0009 | **3.50x** | 7.664 +/- 0.109 | **1.24x** |
| titanic | lp | tv | 1.00e-04 | 5 | `J = 0` | -- | -- | 250 +/- 0 | 1.00x | 0.0197 +/- 0.0018 | 1.00x | 5.930 +/- 0.064 | 1.00x |
| titanic | lp | tv | 1.00e-04 | 5 | `J_a` | 0.3 | 0 | 159 +/- 5 | **1.57x** | 0.0033 +/- 0.0006 | **6.01x** | 5.261 +/- 0.073 | **1.13x** |
| titanic | lp | tv | 1.00e-04 | 5 | `J_s` / `J_g` | 0.3 | 3 | 240 +/- 7 | **1.04x** | 0.0124 +/- 0.0033 | **1.59x** | 3.403 +/- 0.153 | **1.74x** |

