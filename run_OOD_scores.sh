#!/bin/bash


# ============================
#  Configuration
# ============================

BASE_DIR= #Directory with seed folders
DATA_PATH=
TXT_PATH= #Path to list of images
SEEDS=(0 1 2 3 4)  # same seeds as training

# Model checkpoint filename pattern
CKPT_NAME= # .pth file name 

for SEED in "${SEEDS[@]}"; do
    echo "=== Evaluating seed $SEED ==="
    SEED_EXP_DIR=${BASE_DIR}/seed_${SEED}

    python3 OOD_scores.py \
        --save_path $SEED_EXP_DIR/ \
        --state_dict $BASE_DIR/seed_${SEED}/$CKPT_NAME \
        --data_path $DATA_PATH \
        --txt_path $TXT_PATH
done

