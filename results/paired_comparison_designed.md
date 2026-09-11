Paired differences against the J = 0 baseline at the final iteration, designed run (same walker seeds, so differences are paired).
A difference is only meaningful if it is a few standard errors from zero.

| dataset | variant | accuracy difference | error difference |
|---|---|---|---|
| titanic | constant | -0.0005 +/- 0.0006 | -0.151 +/- 0.021 |
| titanic | state | -0.0008 +/- 0.0007 | -0.140 +/- 0.021 |
| titanic | state_nocorr | -0.0009 +/- 0.0007 | -0.139 +/- 0.021 |
| titanic | precond | +0.0002 +/- 0.0007 | -0.252 +/- 0.019 |
| magic | constant | +0.0006 +/- 0.0001 | -0.839 +/- 0.062 |
| magic | state | +0.0005 +/- 0.0001 | -0.815 +/- 0.058 |
| magic | state_nocorr | +0.0005 +/- 0.0001 | -0.815 +/- 0.058 |
| magic | precond | +0.0010 +/- 0.0001 | -1.195 +/- 0.064 |
| breast_cancer | constant | -0.0073 +/- 0.0015 | -0.486 +/- 0.059 |
| breast_cancer | state | -0.0020 +/- 0.0009 | -0.194 +/- 0.021 |
| breast_cancer | state_nocorr | -0.0020 +/- 0.0009 | -0.194 +/- 0.021 |
| breast_cancer | precond | -0.0016 +/- 0.0013 | +0.632 +/- 0.092 |
| spambase | constant | +0.0036 +/- 0.0004 | -2.040 +/- 0.141 |
| spambase | state | -0.0021 +/- 0.0004 | +0.351 +/- 0.030 |
| spambase | state_nocorr | -0.0021 +/- 0.0004 | +0.351 +/- 0.030 |
| spambase | precond | +0.0038 +/- 0.0003 | +9.421 +/- 0.076 |
