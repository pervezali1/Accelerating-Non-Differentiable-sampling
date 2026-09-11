Paired differences against the J = 0 baseline at the final iteration, designed run (same walker seeds, so differences are paired).
A difference is only meaningful if it is a few standard errors from zero.

| dataset | variant | accuracy difference | error difference |
|---|---|---|---|
| titanic | constant | +0.0002 +/- 0.0008 | -0.271 +/- 0.034 |
| titanic | state | +0.0006 +/- 0.0007 | -0.258 +/- 0.033 |
| titanic | state_nocorr | +0.0006 +/- 0.0007 | -0.257 +/- 0.033 |
| titanic | precond | +0.0012 +/- 0.0007 | -0.508 +/- 0.030 |
| magic | constant | +0.0008 +/- 0.0001 | -2.878 +/- 0.074 |
| magic | state | +0.0008 +/- 0.0001 | -2.835 +/- 0.062 |
| magic | state_nocorr | +0.0008 +/- 0.0001 | -2.835 +/- 0.062 |
| magic | precond | +0.0012 +/- 0.0001 | -3.128 +/- 0.072 |
| breast_cancer | constant | -0.0068 +/- 0.0017 | -0.697 +/- 0.044 |
| breast_cancer | state | -0.0022 +/- 0.0012 | -0.503 +/- 0.023 |
| breast_cancer | state_nocorr | -0.0022 +/- 0.0012 | -0.502 +/- 0.023 |
| breast_cancer | precond | -0.0015 +/- 0.0012 | +3.221 +/- 0.104 |
| spambase | constant | +0.0012 +/- 0.0005 | +2.798 +/- 0.197 |
| spambase | state | -0.0006 +/- 0.0004 | -1.151 +/- 0.034 |
| spambase | state_nocorr | -0.0006 +/- 0.0004 | -1.150 +/- 0.034 |
| spambase | precond | +0.0083 +/- 0.0006 | +38.433 +/- 0.105 |
