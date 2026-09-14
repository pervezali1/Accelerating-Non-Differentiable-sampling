# The step size and the gap to the exact posterior

`J = 0`, the paper's batch size and walker count, total simulated time `eta k`
held fixed as `eta` shrinks.  `error` is the whitened distance from the
time-averaged mean (second half of the run) to the exact constrained lasso
posterior's mean, in units of its own standard deviations.

| problem | eta | iterations | error | accuracy, train | accuracy, test |
|---|---|---|---|---|---|
| titanic | 1.00e-04 | 1500 | 0.409 | 0.7797 | 0.7538 |
| titanic | 2.50e-05 | 6000 | 0.412 | 0.7804 | 0.7518 |
| titanic | 6.25e-06 | 24000 | 0.319 | 0.7803 | 0.7510 |
| titanic **exact** | -- | -- | 0 | 0.7808 | 0.7530 |
| magic | 1.00e-04 | 1000 | 21.492 | 0.7829 | 0.7906 |
| magic | 2.50e-05 | 4000 | 6.320 | 0.7822 | 0.7873 |
| magic | 6.25e-06 | 16000 | 1.391 | 0.7839 | 0.7861 |
| magic **exact** | -- | -- | 0 | 0.7850 | 0.7910 |
