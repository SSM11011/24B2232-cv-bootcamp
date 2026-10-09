#!/usr/bin/env bash
# Run the whole pipeline, Task 1 to Task 5, in order. Stops at the first failing step.
#
# Training needs PyTorch with CUDA; analysis needs numpy and matplotlib. On this machine those
# live in different interpreters, so each step names the one it uses:
#   TRAIN_PY      Python with torch      (default: ~/IE643/.venv/bin/python)
#   ANALYSIS_PY   Python with matplotlib (default: python3)
# Smoke test of the full pipeline in a scratch folder, 2 epochs per network:
#   CV_OUTPUT_ROOT=/tmp/cv_smoke CV_EPOCHS=2 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

TRAIN_PY=${TRAIN_PY:-$HOME/IE643/.venv/bin/python}
ANALYSIS_PY=${ANALYSIS_PY:-python3}
LOG_DIR=${CV_OUTPUT_ROOT:-..}/logs
mkdir -p "$LOG_DIR"

step() {
    local python=$1 script=$2
    echo "=== $(date +%H:%M:%S) $script"
    "$python" "$script" 2>&1 | tee "$LOG_DIR/${script%.py}.log"
}

step "$ANALYSIS_PY" task1_noise.py     # noise rate, confusion matrix
step "$TRAIN_PY"    task1_train.py     # clean reference + noisy CE        (2 networks)
step "$ANALYSIS_PY" task2.py           # dynamics, early stopping
step "$TRAIN_PY"    task3_folds.py     # cross-validation runs             (5 networks)
step "$ANALYSIS_PY" task3_scores.py    # noise scores, noise-rate estimate
step "$ANALYSIS_PY" task3_check.py     # AUROC / precision@k (clean labels, measurement only)
step "$TRAIN_PY"    task4_train.py     # GCE, co-teaching, filter-and-retrain (4 networks)
step "$TRAIN_PY"    task5_train.py     # synthetic labels                  (3 networks)
step "$ANALYSIS_PY" task5_check.py     # real vs synthetic (clean labels, measurement only)
echo "=== $(date +%H:%M:%S) done"
