| dataset | variant | step size | acceptance | acc @100 | acc @end | iters to +/-0.005 | mean error | ESS(U) |
|---|---|---|---|---|---|---|---|---|
| titanic | zero | 0.0075 | 0.47 | 0.7823 | 0.7817 | 19 | 0.342 | 1421 |
| titanic | constant | 0.000115 | 0.73 | 0.7558 | 0.7688 | not reached | 4.703 | 221 |
| titanic | state | 0.00649 | 0.54 | 0.7828 | 0.7817 | 22 | 0.361 | 1598 |
| titanic | state_nocorr | 0.00649 | 0.54 | 0.7828 | 0.7816 | 22 | 0.358 | 1618 |
| breast_cancer | zero | 0.00316 | 0.89 | 0.9246 | 0.9675 | 358 | 3.742 | 398 |
| breast_cancer | constant | 0.00205 | 0.80 | 0.9636 | 0.9669 | 83 | 3.920 | 343 |
| breast_cancer | state | 0.00316 | 0.83 | 0.9255 | 0.9673 | 384 | 3.767 | 393 |
| breast_cancer | state_nocorr | 0.00316 | 0.83 | 0.9255 | 0.9675 | 391 | 3.764 | 400 |
