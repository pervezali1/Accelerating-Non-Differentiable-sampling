# Is the gate open where the chain lives?

`J_s(w) = s(w) J_a` with `s(w) = 1 / (1 + exp((r(w) - R) / W))`,
`R = 0.75 r0`, `W = 0.1 r0`, `r0` = whitened distance from `w = 0` to the
warm-up mean.  Radii are measured in the gate's own metric: centred at the
warm-up mean `m`, in the warm-up inverse covariance `H`.  If that geometry
matched the posterior the bulk would sit at radius `sqrt(d)`, so `sqrt(d)`
is the column to compare the measured bulk radius against -- a bulk radius
far above it means the warm-up never reached the bulk, and then `r0` is not
the distance to the bulk at all.  The gate is open where the chain lives
only if `r0` is comfortably larger than the bulk radius.  Gate values are
averaged over draws from the reference posterior.

| dataset | `d` | `r0` (start to bulk) | `sqrt(d)` | bulk radius (median) | `R` | `W` | mean `s` in the bulk |
|---|---|---|---|---|---|---|---|
| Titanic (survival) | 10 | 13.90 | 3.16 | 3.07 | 10.43 | 1.39 | 0.994 |
| MAGIC Gamma Telescope | 11 | 58.67 | 3.32 | 3.13 | 44.00 | 5.87 | 0.999 |
| Breast Cancer Wisconsin | 31 | 1.38 | 5.57 | 36.42 | 1.04 | 0.14 | 3.15e-46 |
| Spambase | 58 | 0.94 | 7.62 | 34.93 | 0.70 | 0.09 | 7.67e-104 |
