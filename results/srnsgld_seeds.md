# The same comparison over several seeds

Whitened error of the running posterior mean at iteration 200, 32 walkers,
rotation strength rho = 2, 6 seeds, each seed shared by all
three fields (common random numbers).  `ratio` is the `J = 0` error over this
field's, seed by seed, quoted as mean +/- standard error.

## Titanic (survival)

| field | error | ratio to `J = 0` |
|---|---|---|
| `zero` | 0.6282 +/- 0.0458 | 1.000 +/- 0.000 |
| `constant` | 0.4266 +/- 0.0462 | 1.528 +/- 0.119 |
| `state` | 0.5192 +/- 0.0443 | 1.217 +/- 0.038 |

## MAGIC Gamma Telescope

| field | error | ratio to `J = 0` |
|---|---|---|
| `zero` | 23.3483 +/- 0.2914 | 1.000 +/- 0.000 |
| `constant` | 11.4612 +/- 0.1378 | 2.038 +/- 0.019 |
| `state` | 21.9205 +/- 0.2770 | 1.065 +/- 0.003 |

