# Choosing the lasso strength

`lam = c n_train`, `delta = 0.5 / lam`.  `|x*|` is the norm of the constrained
lasso MAP and `on boundary` says whether the constraint is still active there;
`zeros` counts coordinates within 0.01 of zero; `clock` is `exp(U - U_0)` at that
MAP; `eta lam` is the step size times the lasso strength, which is roughly the
jitter the explicit scheme leaves on a coordinate sitting at the kink.

| problem | set | c | lam | delta | eta lam | \|x*\| | on boundary | zeros | clock |
|---|---|---|---|---|---|---|---|---|---|
| synthetic **used** | ball | 0.01 | 16.0 | 0.03125 | 0.0016 | 1.000 | yes | 0/3 | 0.848 |
| synthetic | ball | 0.03 | 48.0 | 0.01042 | 0.0048 | 0.937 | no | 0/3 | 0.830 |
| synthetic | ball | 0.1 | 160.0 | 0.00313 | 0.0160 | 0.592 | no | 1/3 | 0.695 |
| synthetic | ball | 0.3 | 480.0 | 0.00104 | 0.0480 | 0.074 | no | 2/3 | 0.530 |
| synthetic **used** | lp | 0.01 | 16.0 | 0.03125 | 0.0016 | 1.073 | no | 0/3 | 0.853 |
| synthetic | lp | 0.03 | 48.0 | 0.01042 | 0.0048 | 0.937 | no | 0/3 | 0.830 |
| synthetic | lp | 0.1 | 160.0 | 0.00313 | 0.0160 | 0.592 | no | 1/3 | 0.695 |
| synthetic | lp | 0.3 | 480.0 | 0.00104 | 0.0480 | 0.074 | no | 2/3 | 0.530 |
| magic **used** | ball | 0.01 | 152.2 | 0.00329 | 0.0152 | 1.414 | yes | 2/9 | 0.446 |
| magic | ball | 0.03 | 456.5 | 0.00110 | 0.0456 | 1.262 | no | 6/9 | 0.181 |
| magic | ball | 0.1 | 1521.6 | 0.00033 | 0.1522 | 0.525 | no | 7/9 | 0.112 |
| magic | ball | 0.3 | 4564.8 | 0.00011 | 0.4565 | 0.000 | no | 9/9 | 0.034 |
| magic **used** | lp | 0.01 | 152.2 | 0.00329 | 0.0152 | 1.756 | no | 4/9 | 0.300 |
| magic | lp | 0.03 | 456.5 | 0.00110 | 0.0456 | 1.262 | no | 6/9 | 0.181 |
| magic | lp | 0.1 | 1521.6 | 0.00033 | 0.1522 | 0.525 | no | 7/9 | 0.112 |
| magic | lp | 0.3 | 4564.8 | 0.00011 | 0.4565 | 0.000 | no | 9/9 | 0.034 |
| titanic **used** | ball | 0.01 | 7.1 | 0.07013 | 0.0007 | 1.414 | yes | 0/9 | 0.406 |
| titanic | ball | 0.03 | 21.4 | 0.02338 | 0.0021 | 1.145 | no | 1/9 | 0.285 |
| titanic | ball | 0.1 | 71.3 | 0.00701 | 0.0071 | 0.724 | no | 6/9 | 0.083 |
| titanic | ball | 0.3 | 213.9 | 0.00234 | 0.0214 | 0.005 | no | 9/9 | 0.036 |
| titanic **used** | lp | 0.01 | 7.1 | 0.07013 | 0.0007 | 1.446 | no | 0/9 | 0.416 |
| titanic | lp | 0.03 | 21.4 | 0.02338 | 0.0021 | 1.145 | no | 1/9 | 0.285 |
| titanic | lp | 0.1 | 71.3 | 0.00701 | 0.0071 | 0.724 | no | 6/9 | 0.083 |
| titanic | lp | 0.3 | 213.9 | 0.00234 | 0.0214 | 0.005 | no | 9/9 | 0.036 |
