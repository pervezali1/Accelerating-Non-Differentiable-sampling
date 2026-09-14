# Ridge against lasso

Same protocol on both targets: 200 iterations, 24 walkers from w = 0, common
seeds, three fields, each at its own step size. The ridge target is smooth and
sampled with the plain scheme; the lasso target is non-differentiable and sampled
with anchored Langevin. The two priors give different posteriors, so compare each
rotation with the reversible baseline inside its own target, not across targets.

| dataset | prior | field | loss gap | iters to loss band | speedup over J = 0 | mean error | error ratio to J = 0 | step vs J = 0 |
|---|---|---|---|---|---|---|---|---|
| Titanic (survival) | ridge | J = 0 | 0.0013 | 74 | 1.0 | 0.932 | 1.0 | 1.0 |
| Titanic (survival) | ridge | constant J_a | 0.0002 | 19 | 3.89 | 0.58 | 1.61 | 1.75 |
| Titanic (survival) | ridge | gated J_s | 0.0002 | 21 | 3.52 | 0.582 | 1.6 | 1.75 |
| Titanic (survival) | lasso | J = 0 | 0.002 | 97 | 1.0 | 1.039 | 1.0 | 1.0 |
| Titanic (survival) | lasso | constant J_a | 0.0008 | 37 | 2.62 | 0.629 | 1.65 | 1.81 |
| Titanic (survival) | lasso | gated J_s | 0.001 | 42 | 2.31 | 0.633 | 1.64 | 1.81 |
| MAGIC Gamma Telescope | ridge | J = 0 | 0.0002 | 60 | 1.0 | 4.016 | 1.0 | 1.0 |
| MAGIC Gamma Telescope | ridge | constant J_a | 0.0 | 25 | 2.4 | 1.117 | 3.6 | 1.97 |
| MAGIC Gamma Telescope | ridge | gated J_s | -0.0001 | 24 | 2.5 | 1.185 | 3.39 | 1.97 |
| MAGIC Gamma Telescope | lasso | J = 0 | 0.0007 | 84 | 1.0 | 4.915 | 1.0 | 1.0 |
| MAGIC Gamma Telescope | lasso | constant J_a | -0.0001 | 39 | 2.15 | 1.859 | 2.64 | 1.96 |
| MAGIC Gamma Telescope | lasso | gated J_s | -0.0001 | 38 | 2.21 | 1.787 | 2.75 | 1.96 |

## The lasso as a minimisation

scikit-learn with penalty='l1' and C = 1/lambda = 0.2 minimises the same
objective U, so its solution is the MAP of the posterior that was sampled.

| index | Titanic (survival) | MAGIC Gamma Telescope |
|---|---|---|
| U at the lasso minimiser | 288.34 | 6102.3 |
| U at the posterior mean | 288.52 | 6102.39 |
| exact zeros, minimiser | 1.0 | 1.0 |
| exact zeros, posterior mean | 0.0 | 0.0 |
| coefficients | 9.0 | 10.0 |
| ||w||_1, minimiser | 3.105 | 3.78 |
| ||w||_1, posterior mean | 3.286 | 3.799 |
| test accuracy, minimiser | 0.7724 | 0.7948 |
| test accuracy, posterior | 0.7724 | 0.795 |
| test log-loss, minimiser | 0.4631 | 0.4586 |
| test log-loss, posterior | 0.4608 | 0.4586 |
