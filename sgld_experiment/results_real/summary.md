# Real data: reversible vs non-reversible anchored Langevin

Final runs: R = 100 shared-randomness replicates, held-out confirmation on four extra sampler seeds (R = 100 each) at the iteration where the search-run gap peaked.  Search rounds: R = 40, 1000 iterations (2000 for the raw-scale runs), MAGIC subsampled to 4000 training rows.

## Final runs

| run | n_train | reference acc. | REV / NR at the peak | peak gap (it.) | held-out pooled gap | t | gap at the end |
|---|---|---|---|---|---|---|---|
| Titanic, L1-smooth ball | 712 | 0.782 | 0.540 / 0.730 | +0.190 (140) | +0.174 | +22.3 | -0.0042 |
| Titanic, unit ball | 712 | 0.782 | 0.568 / 0.709 | +0.140 (110) | +0.143 | +19.6 | -0.0015 |
| MAGIC telescope, L1-smooth ball | 15216 | 0.790 | 0.550 / 0.737 | +0.188 (20) | +0.127 | +29.7 | -0.0001 |
| MAGIC telescope, unit ball | 15216 | 0.790 | 0.538 / 0.701 | +0.162 (40) | +0.157 | +29.0 | -0.0010 |

## Search: best valid configuration per preprocessing ladder (projection rate < 0.05)

| data set | ladder | best peak gap (it.) | t | REV / NR | configurations |
|---|---|---|---|---|---|
| titanic | standard scaling, natural order | +0.018 (70) | +2.3 | 0.599 / 0.618 | 30 |
| titanic | standard scaling, importance order | +0.014 (40) | +2.2 | 0.528 / 0.542 | 6 |
| titanic | equalised coefficients | +0.018 (30) | +1.4 | 0.624 / 0.643 | 30 |
| titanic | block reparametrisation, L1 | +0.233 (50) | +10.2 | 0.515 / 0.748 | 64 |
| titanic | standard scaling, unit ball | +0.024 (200) | +2.6 | 0.654 / 0.678 | 11 |
| titanic | block reparametrisation, unit ball | +0.119 (140) | +5.7 | 0.588 / 0.707 | 26 |
| magic | standard scaling, natural order | +0.014 (40) | +1.7 | 0.610 / 0.624 | 30 |
| magic | standard scaling, importance order | +0.017 (60) | +2.1 | 0.638 / 0.655 | 6 |
| magic | equalised coefficients | +0.008 (10) | +1.0 | 0.612 / 0.620 | 14 |
| magic | raw (unstandardised) scales | +0.035 (560) | +4.0 | 0.587 / 0.623 | 6 |
| magic | block reparametrisation, L1 | +0.189 (20) | +9.8 | 0.546 / 0.735 | 58 |
| magic | standard scaling, unit ball | +0.011 (10) | +1.2 | 0.528 / 0.539 | 11 |
| magic | block reparametrisation, unit ball | +0.131 (60) | +7.4 | 0.573 / 0.704 | 21 |
