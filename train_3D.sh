#!/bin/bash


# Define variables
BASE_EXP_DIR= # saving directory
DATA_PATH= #path to images
TRAIN_TXT= #path to list of images for training

SEEDS=(0 1 2 3 4)  # list of seeds you want to use

# Loop over seeds
for SEED in "${SEEDS[@]}"; do
    echo "=== Running seed $SEED ==="
    EXPERIMENT_DIR=${BASE_EXP_DIR}/seed_${SEED}
    mkdir -p $EXPERIMENT_DIR

    python3 train_3DVAE_freq_2gauss_adjustB.py \
        --experiment $EXPERIMENT_DIR \
        --num_epoch 10 \
        --seed_val $SEED \
        --data_path $DATA_PATH \
        --train_txt $TRAIN_TXT 
done