| dataset | variant | step size | acceptance | acc @100 | acc @end | iters to +/-0.005 | mean error | ESS(U) |
|---|---|---|---|---|---|---|---|---|
| titanic | zero | 0.649 | 0.48 | 0.7824 | 0.7822 | 3 | 0.264 | 1799 |
| titanic | constant | 0.00487 | 0.75 | 0.7703 | 0.7812 | 478 | 5.084 | 209 |
| titanic | state | 0.237 | 0.52 | 0.7831 | 0.7825 | 13 | 0.436 | 1423 |
| titanic | state_nocorr | 0.237 | 0.52 | 0.7830 | 0.7826 | 13 | 0.433 | 1456 |
| magic | zero | 0.274 | 0.55 | 0.7948 | 0.7951 | 9 | 0.594 | 1676 |
| magic | constant | 0.000205 | 0.54 | 0.7584 | 0.7758 | not reached | 56.886 | 93 |
| magic | state | 0.178 | 0.52 | 0.7947 | 0.7951 | 7 | 0.939 | 1175 |
| magic | state_nocorr | 0.178 | 0.52 | 0.7947 | 0.7951 | 7 | 0.943 | 1152 |
| breast_cancer | zero | 0.01 | 0.93 | 0.9465 | 0.9627 | 438 | 3.269 | 267 |
| breast_cancer | constant | 0.00649 | 0.85 | 0.9596 | 0.9649 | 110 | 3.377 | 244 |
| breast_cancer | state | 0.01 | 0.82 | 0.9458 | 0.9636 | 438 | 3.314 | 271 |
| breast_cancer | state_nocorr | 0.01 | 0.82 | 0.9460 | 0.9636 | 438 | 3.311 | 274 |
| spambase | zero | 0.0178 | 0.66 | 0.6312 | 0.9239 | not reached | 12.838 | 216 |
| spambase | constant | 2.05e-05 | 0.61 | 0.8291 | 0.8850 | not reached | 26.004 | 93 |
| spambase | state | 0.0178 | 0.57 | 0.6842 | 0.9380 | 452 | 11.652 | 214 |
| spambase | state_nocorr | 0.0178 | 0.57 | 0.6842 | 0.9379 | 449 | 11.648 | 217 |
