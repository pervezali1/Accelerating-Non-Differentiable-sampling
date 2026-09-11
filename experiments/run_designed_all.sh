#!/usr/bin/env bash
# Designed-rotation experiment: one process per dataset (4 cores assumed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results figures
ARGS=${ARGS:-""}
for ds in titanic breast_cancer magic spambase; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 experiments/run_designed.py --datasets "$ds" $ARGS >"logs/designed_${ds}.log" 2>&1 &
done
wait
echo "all designed runs finished"
