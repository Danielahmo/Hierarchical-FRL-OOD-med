#!/bin/bash

DATA_PATH=
TXT_PATH=
BASE_EXP_DIR=
SEEDS=(0 1 2 3 4)  # list of seeds you want to use


for SEED in "${SEEDS[@]}"; do
    echo "=== Evaluating seed $SEED ==="    
    SAVE_DIR=${BASE_EXP_DIR}/seed_${SEED}/
    python3 OOD_test_3DVAE_mse.py \
        --data_path $DATA_PATH\
        --test_path $TXT_PATH\
        --save_path  $SAVE_DIR \
        --state_dict $BASE_EXP_DIR/seed_${SEED}/net_ngf_32_nz_100_epoch.pth \

done 
      