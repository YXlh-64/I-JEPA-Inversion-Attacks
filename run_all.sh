#!/usr/bin/env bash
set -euo pipefail

# Simple end-to-end runner: prepare dataset -> train (DO/DB/DMB) -> run attacks
# Customize via env vars before running, e.g. EPOCHS_DO=5 ./run_all.sh

PY=${PY-python}
PREPARED_ROOT=${PREPARED_ROOT-prepared_data}

# Dataset limits
TRAIN_LIMIT=${TRAIN_LIMIT-5000}
TEST_LIMIT=${TEST_LIMIT-100}

# Training knobs
EPOCHS_DO=${EPOCHS_DO-10}
EPOCHS_DB=${EPOCHS_DB-10}
EPOCHS_DMB=${EPOCHS_DMB-5}
BATCH_DO=${BATCH_DO-32}
BATCH_DB=${BATCH_DB-8}
BATCH_DMB=${BATCH_DMB-4}

# DMB mode: set to 1 to skip full diffusion and only VAE-decode predicted latents (faster)
DMB_PREDICT_LATENTS=${DMB_PREDICT_LATENTS-1}
DMB_STEPS=${DMB_STEPS-30}

log() { echo -e "\n==== $* ===="; }

# 1) Prepare dataset (CIFAR-10 -> IJepa embeddings + SD VAE latents)
log "Preparing dataset (train=${TRAIN_LIMIT}, test=${TEST_LIMIT})"
"$PY" scripts/prepare_dataset.py \
  --train-limit "$TRAIN_LIMIT" \
  --test-limit "$TEST_LIMIT"

# 2) Train DO
log "Training DO"
"$PY" scripts/train_do.py \
  --prepared-root "$PREPARED_ROOT" \
  --epochs "$EPOCHS_DO" \
  --batch-size "$BATCH_DO" \
  --mixed-precision

# 3) Train DB
log "Training DB"
"$PY" scripts/train_db.py \
  --prepared-root "$PREPARED_ROOT" \
  --epochs "$EPOCHS_DB" \
  --batch-size "$BATCH_DB" \
  --mixed-precision

# 4) Train DMB
log "Training DMB"
if [[ "$DMB_PREDICT_LATENTS" == "1" ]]; then
  DMB_FLAGS=(--predict-latents)
else
  DMB_FLAGS=(--num-inference-steps "$DMB_STEPS")
fi
"$PY" scripts/train_dmb.py \
  --prepared-root "$PREPARED_ROOT" \
  --epochs "$EPOCHS_DMB" \
  --batch-size "$BATCH_DMB" \
  --mixed-precision \
  "${DMB_FLAGS[@]}"

# 5) Run attacks and save comparison grids under results/
log "Running DO attack"
"$PY" attacks/attack_do.py --prepared-root "$PREPARED_ROOT" --split test --output-dir results

log "Running DB attack"
"$PY" attacks/attack_db.py --prepared-root "$PREPARED_ROOT" --split test --output-dir results

log "Running DMB attack"
if [[ "$DMB_PREDICT_LATENTS" == "1" ]]; then
  DMB_ATTACK_FLAGS=(--predict-latents)
else
  DMB_ATTACK_FLAGS=(--num-inference-steps "$DMB_STEPS")
fi
"$PY" attacks/attack_dmb.py --prepared-root "$PREPARED_ROOT" --split test --output-dir results "${DMB_ATTACK_FLAGS[@]}"

log "All done. Checkpoints saved in checkpoints_*/ and results in results/"
