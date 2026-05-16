#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PROGRESS_EVERY_BATCHES="${PROGRESS_EVERY_BATCHES:-25}"
LAMBDA_MATRIX="${LAMBDA_MATRIX:-0.05}"
MIN_MATRIX_PAIRS="${MIN_MATRIX_PAIRS:-16}"
HEAD_EPOCHS="${HEAD_EPOCHS:-5}"
LAYER4_EPOCHS="${LAYER4_EPOCHS:-10}"
REAL_OUTPUT_DIR="${REAL_OUTPUT_DIR:-outputs/joint_matrix_resnet50}"
SHUFFLED_OUTPUT_DIR="${SHUFFLED_OUTPUT_DIR:-outputs/joint_matrix_resnet50_shuffled}"

EXTRA_ARGS=()

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  HEAD_EPOCHS="${SMOKE_HEAD_EPOCHS:-1}"
  LAYER4_EPOCHS="${SMOKE_LAYER4_EPOCHS:-1}"
  EXTRA_ARGS+=(
    --max-train-batches "${SMOKE_MAX_TRAIN_BATCHES:-100}"
    --max-eval-batches "${SMOKE_MAX_EVAL_BATCHES:-30}"
  )
  REAL_OUTPUT_DIR="${SMOKE_REAL_OUTPUT_DIR:-outputs/joint_matrix_resnet50_smoke}"
  SHUFFLED_OUTPUT_DIR="${SMOKE_SHUFFLED_OUTPUT_DIR:-outputs/joint_matrix_resnet50_shuffled_smoke}"
fi

if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi

COMMON_ARGS=(
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --head-epochs "$HEAD_EPOCHS"
  --layer4-epochs "$LAYER4_EPOCHS"
  --lambda-matrix "$LAMBDA_MATRIX"
  --min-matrix-pairs "$MIN_MATRIX_PAIRS"
  --progress-every-batches "$PROGRESS_EVERY_BATCHES"
  "${EXTRA_ARGS[@]}"
)

echo "Running joint matrix model"
echo "  batch size: $BATCH_SIZE"
echo "  workers: $NUM_WORKERS"
echo "  head epochs: $HEAD_EPOCHS"
echo "  layer4 epochs: $LAYER4_EPOCHS"
echo "  lambda matrix: $LAMBDA_MATRIX"
echo "  output: $REAL_OUTPUT_DIR"
echo "  resume: ${RESUME:-0}"

"$PYTHON_BIN" scripts/15_train_resnet50_joint_matrix.py \
  "${COMMON_ARGS[@]}" \
  --output-dir "$REAL_OUTPUT_DIR"

echo "Running shuffled joint matrix control"
echo "  output: $SHUFFLED_OUTPUT_DIR"

"$PYTHON_BIN" scripts/15_train_resnet50_joint_matrix.py \
  "${COMMON_ARGS[@]}" \
  --shuffle-human-matrix \
  --output-dir "$SHUFFLED_OUTPUT_DIR"

echo "Done."
echo "Real metrics: $REAL_OUTPUT_DIR/metrics.json"
echo "Shuffled metrics: $SHUFFLED_OUTPUT_DIR/metrics.json"
