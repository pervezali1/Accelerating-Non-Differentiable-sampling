# What the clock is worth

`error` is the whitened distance from the time-averaged mean to the exact
constrained lasso posterior's mean.  `anchored` keeps the clock
`a = exp(U - U_0)`, so the invariant law is the kinked one; `anchor only` drops
it, so the invariant law is the smoothed one.  `near-zero error` is the largest
absolute error over the coordinates whose exact posterior mean is within
0.05 of zero -- the coordinates the lasso is acting on, where the two
laws differ most.

| problem | field | error, anchored | error, anchor only | near-zero coords | near-zero error, anchored | near-zero error, anchor only |
|---|---|---|---|---|---|---|
| synthetic | `zero` | 0.615 | 0.711 | 0/3 | -- | -- |
| synthetic | `constant` | 0.614 | 0.709 | 0/3 | -- | -- |
| synthetic | `sublevel` | 1.551 | 1.988 | 0/3 | -- | -- |
| magic | `zero` | 5.925 | 6.416 | 4/9 | 0.0665 | 0.0708 |
| magic | `constant` | 5.771 | 6.275 | 4/9 | 0.0637 | 0.0681 |
| magic | `sublevel` | 5.347 | 5.495 | 4/9 | 0.0563 | 0.0566 |
| titanic | `zero` | 0.545 | 0.377 | 0/9 | -- | -- |
| titanic | `constant` | 0.353 | 0.558 | 0/9 | -- | -- |
| titanic | `sublevel` | 0.366 | 1.900 | 0/9 | -- | -- |
