#!/bin/bash#
=============================================================================
# PSM Dataset - Training & Evaluation Script# =============================================================================
# Usage: bash scripts/run_psm.sh

DATASET="PSM"
DATA_PATH="../datasets/PSM"
INPUT_C=25
OUTPUT_C=25

# Hyperparameters
N_MEMORY=5
N_MEMORY_LONG=1
LONG_TERM_MULTIPLIER=2
BATCH_SIZE=16

# Common settings
LAMBD=0.01
DOWNSAMPLE_METHOD="linear_interpolation"
UPSAMPLE_METHOD="linear_interpolation"
DEVICE="cuda:0"
NUM_EPOCHS=100
FEATURE_INDICES="0 1 2 3 4 5 6"
K=3
RUN_NAME="default"

echo "============================================================"
echo "PSM Dataset Training"
echo "============================================================"
echo "N_MEMORY: $N_MEMORY"
echo "N_MEMORY_LONG: $N_MEMORY_LONG"
echo "LONG_TERM_MULTIPLIER: $LONG_TERM_MULTIPLIER"
echo "BATCH_SIZE: $BATCH_SIZE"
echo ""

# ----- First Step -----
echo "[1/3] First Step Training..."
python main.py \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --mode train_first_step \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --input_c $INPUT_C \
    --output_c $OUTPUT_C \
    --n_memory $N_MEMORY \
    --n_memory_long $N_MEMORY_LONG \
    --lambd $LAMBD \
    --lr 1e-4 \
    --phase_type first_train \
    --memory_initial False \
    --run_name $RUN_NAME \
    --long_term_multiplier $LONG_TERM_MULTIPLIER \
    --downsample_method $DOWNSAMPLE_METHOD \
    --upsample_method $UPSAMPLE_METHOD \
    --device $DEVICE \
    --feature_indices $FEATURE_INDICES \
    --k $K

# ----- Second Step -----
echo "[2/3] Second Step Training..."
python main.py \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --mode train_second_step \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --input_c $INPUT_C \
    --output_c $OUTPUT_C \
    --n_memory $N_MEMORY \
    --n_memory_long $N_MEMORY_LONG \
    --lambd $LAMBD \
    --lr 5e-5 \
    --phase_type second_train \
    --memory_initial True \
    --run_name $RUN_NAME \
    --long_term_multiplier $LONG_TERM_MULTIPLIER \
    --downsample_method $DOWNSAMPLE_METHOD \
    --upsample_method $UPSAMPLE_METHOD \
    --device $DEVICE \
    --feature_indices $FEATURE_INDICES \
    --k $K

# ----- Test -----
echo "[3/3] Testing..."
python main.py \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --mode test \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --input_c $INPUT_C \
    --output_c $OUTPUT_C \
    --n_memory $N_MEMORY \
    --n_memory_long $N_MEMORY_LONG \
    --phase_type test \
    --memory_initial False \
    --run_name $RUN_NAME \
    --long_term_multiplier $LONG_TERM_MULTIPLIER \
    --downsample_method $DOWNSAMPLE_METHOD \
    --upsample_method $UPSAMPLE_METHOD \
    --device $DEVICE \
    --feature_indices $FEATURE_INDICES \
    --k $K

echo "============================================================"
echo "PSM Training Completed!"
echo "============================================================"