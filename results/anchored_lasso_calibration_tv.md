# Choosing the penalty strength (tv)

`lam = c n_train`, `delta = 0.5 / lam`.  `|x*|` is the norm of the constrained
lasso MAP and `on boundary` says whether the constraint is still active there;
`zeros` counts coordinates within 0.01 of zero; `clock` is `exp(U - U_0)` at that
MAP; `eta lam` is the step size times the lasso strength, which is roughly the
jitter the explicit scheme leaves on a coordinate sitting at the kink.

| problem | set | c | lam | delta | eta lam | \|x*\| | on boundary | zeros | clock |
|---|---|---|---|---|---|---|---|---|---|
| synthetic **used** | ball | 0.01 | 16.0 | 0.04688 | 0.0016 | 1.000 | yes | 0/3 | 0.958 |
| synthetic | ball | 0.03 | 48.0 | 0.01562 | 0.0048 | 0.872 | no | 1/3 | 0.981 |
| synthetic | ball | 0.1 | 160.0 | 0.00469 | 0.0160 | 0.541 | no | 0/3 | 0.740 |
| synthetic | ball | 0.3 | 480.0 | 0.00156 | 0.0480 | 0.194 | no | 0/3 | 0.460 |
| synthetic **used** | lp | 0.01 | 16.0 | 0.04688 | 0.0016 | 1.044 | no | 0/3 | 0.960 |
| synthetic | lp | 0.03 | 48.0 | 0.01562 | 0.0048 | 0.872 | no | 1/3 | 0.981 |
| synthetic | lp | 0.1 | 160.0 | 0.00469 | 0.0160 | 0.541 | no | 0/3 | 0.740 |
| synthetic | lp | 0.3 | 480.0 | 0.00156 | 0.0480 | 0.194 | no | 0/3 | 0.460 |
| magic **used** | ball | 0.01 | 152.2 | 0.00370 | 0.0152 | 1.414 | yes | 1/9 | 0.818 |
| magic | ball | 0.03 | 456.5 | 0.00123 | 0.0456 | 1.139 | no | 0/9 | 0.343 |
| magic | ball | 0.1 | 1521.6 | 0.00037 | 0.1522 | 0.619 | no | 0/9 | 0.207 |
| magic | ball | 0.3 | 4564.8 | 0.00012 | 0.4565 | 0.180 | no | 0/9 | 0.084 |
| magic **used** | lp | 0.01 | 152.2 | 0.00370 | 0.0152 | 1.528 | no | 0/9 | 0.802 |
| magic | lp | 0.03 | 456.5 | 0.00123 | 0.0456 | 1.139 | no | 0/9 | 0.343 |
| magic | lp | 0.1 | 1521.6 | 0.00037 | 0.1522 | 0.619 | no | 0/9 | 0.207 |
| magic | lp | 0.3 | 4564.8 | 0.00012 | 0.4565 | 0.180 | no | 0/9 | 0.084 |
| titanic **used** | ball | 0.01 | 7.1 | 0.07889 | 0.0007 | 1.414 | yes | 0/9 | 0.319 |
| titanic | ball | 0.03 | 21.4 | 0.02630 | 0.0021 | 1.109 | no | 1/9 | 0.302 |
| titanic | ball | 0.1 | 71.3 | 0.00789 | 0.0071 | 0.740 | no | 0/9 | 0.099 |
| titanic | ball | 0.3 | 213.9 | 0.00263 | 0.0214 | 0.660 | no | 0/9 | 0.089 |
| titanic **used** | lp | 0.01 | 7.1 | 0.07889 | 0.0007 | 1.437 | no | 0/9 | 0.318 |
| titanic | lp | 0.03 | 21.4 | 0.02630 | 0.0021 | 1.109 | no | 1/9 | 0.302 |
| titanic | lp | 0.1 | 71.3 | 0.00789 | 0.0071 | 0.740 | no | 0/9 | 0.099 |
| titanic | lp | 0.3 | 213.9 | 0.00263 | 0.0214 | 0.660 | no | 0/9 | 0.089 |
