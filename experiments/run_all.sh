#!/usr/bin/env bash
# Run the four-dataset experiment, one process per dataset (4 cores assumed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results figures
ARGS=${ARGS:-"--n-iter 600 --n-walkers 32 --alpha 1.0 --geometry warmup"}
for ds in titanic breast_cancer magic spambase; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 experiments/run_accuracy.py --datasets "$ds" $ARGS >"logs/${ds}.log" 2>&1 &
done
wait
echo "all datasets finished"
