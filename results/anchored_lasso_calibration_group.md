# Choosing the penalty strength (group)

`lam = c n_train`, `delta = 0.5 / lam`.  `|x*|` is the norm of the constrained
lasso MAP and `on boundary` says whether the constraint is still active there;
`zeros` counts coordinates within 0.01 of zero; `clock` is `exp(U - U_0)` at that
MAP; `eta lam` is the step size times the lasso strength, which is roughly the
jitter the explicit scheme leaves on a coordinate sitting at the kink.

| problem | set | c | lam | delta | eta lam | \|x*\| | on boundary | zeros | clock |
|---|---|---|---|---|---|---|---|---|---|
| synthetic **used** | ball | 0.01 | 16.0 | 0.09375 | 0.0016 | 1.000 | yes | 0/3 | 0.932 |
| synthetic | ball | 0.03 | 48.0 | 0.03125 | 0.0048 | 0.993 | no | 0/3 | 0.977 |
| synthetic | ball | 0.1 | 160.0 | 0.00937 | 0.0160 | 0.712 | no | 0/3 | 0.990 |
| synthetic | ball | 0.3 | 480.0 | 0.00313 | 0.0480 | 0.206 | no | 0/3 | 0.989 |
| synthetic **used** | lp | 0.01 | 16.0 | 0.09375 | 0.0016 | 1.097 | no | 0/3 | 0.938 |
| synthetic | lp | 0.03 | 48.0 | 0.03125 | 0.0048 | 0.993 | no | 0/3 | 0.977 |
| synthetic | lp | 0.1 | 160.0 | 0.00937 | 0.0160 | 0.712 | no | 0/3 | 0.990 |
| synthetic | lp | 0.3 | 480.0 | 0.00313 | 0.0480 | 0.206 | no | 0/3 | 0.989 |
| magic **used** | ball | 0.01 | 152.2 | 0.00986 | 0.0152 | 1.414 | yes | 1/9 | 0.968 |
| magic | ball | 0.03 | 456.5 | 0.00329 | 0.0456 | 1.217 | no | 1/9 | 0.984 |
| magic | ball | 0.1 | 1521.6 | 0.00099 | 0.1522 | 0.598 | no | 4/9 | 0.618 |
| magic | ball | 0.3 | 4564.8 | 0.00033 | 0.4565 | 0.000 | no | 9/9 | 0.109 |
| magic **used** | lp | 0.01 | 152.2 | 0.00986 | 0.0152 | 1.735 | no | 1/9 | 0.975 |
| magic | lp | 0.03 | 456.5 | 0.00329 | 0.0456 | 1.217 | no | 1/9 | 0.984 |
| magic | lp | 0.1 | 1521.6 | 0.00099 | 0.1522 | 0.598 | no | 4/9 | 0.618 |
| magic | lp | 0.3 | 4564.8 | 0.00033 | 0.4565 | 0.000 | no | 9/9 | 0.109 |
| titanic **used** | ball | 0.01 | 7.1 | 0.21038 | 0.0007 | 1.414 | yes | 0/9 | 0.463 |
| titanic | ball | 0.03 | 21.4 | 0.07013 | 0.0021 | 1.260 | no | 0/9 | 0.708 |
| titanic | ball | 0.1 | 71.3 | 0.02104 | 0.0071 | 0.753 | no | 3/9 | 0.657 |
| titanic | ball | 0.3 | 213.9 | 0.00701 | 0.0214 | 0.016 | no | 8/9 | 0.145 |
| titanic **used** | lp | 0.01 | 7.1 | 0.21038 | 0.0007 | 1.540 | no | 0/9 | 0.491 |
| titanic | lp | 0.03 | 21.4 | 0.07013 | 0.0021 | 1.260 | no | 0/9 | 0.708 |
| titanic | lp | 0.1 | 71.3 | 0.02104 | 0.0071 | 0.753 | no | 3/9 | 0.657 |
| titanic | lp | 0.3 | 213.9 | 0.00701 | 0.0214 | 0.016 | no | 8/9 | 0.145 |
