#!/bin/bash

# AMTL Enhanced Training Script for NEREL Dataset
# Usage: ./train_amtl.sh [baseline|amtl|amtl-only|sampling-only|full]

set -e  # Exit on error

# Configuration
DATA_DIR="${DATA_DIR:-$HOME/data/relations/NEREL/naa-docred-split}"
SAVE_BASE="${SAVE_BASE:-./logs}"
MODEL_NAME="ai-forever/ruRoberta-large"
TRANSFORMER_TYPE="roberta"

# Training hyperparameters (optimized for RoBERTa-large on NEREL)
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
NUM_LABELS=2
LR_TRANSFORMER=3e-5
MAX_GRAD_NORM=1.0
EVI_THRESH=0.2
EVI_LAMBDA=0.05
WARMUP_RATIO=0.06
NUM_EPOCHS=160
SEED=7777
NUM_CLASS=48
MAX_SEQ_LENGTH=1460
MAX_SENT_NUM=101

# Parse command-line argument
MODE="${1:-full}"

case "$MODE" in
    baseline)
        EXPERIMENT_NAME="nerel-baseline"
        EXTRA_ARGS=""
        echo "=== Training BASELINE model (ATLoss) ==="
        ;;
    amtl-only)
        EXPERIMENT_NAME="nerel-amtl-only"
        EXTRA_ARGS="--use_amtl --use_effective_number"
        echo "=== Training with AMTL ONLY ==="
        ;;
    sampling-only)
        EXPERIMENT_NAME="nerel-sampling-only"
        EXTRA_ARGS="--use_negative_sampling --neg_pos_ratio 3.0"
        echo "=== Training with NEGATIVE SAMPLING ONLY ==="
        ;;
    amtl)
        EXPERIMENT_NAME="nerel-amtl-sampling"
        EXTRA_ARGS="--use_amtl --use_effective_number --use_negative_sampling --neg_pos_ratio 3.0"
        echo "=== Training with AMTL + SAMPLING ==="
        ;;
    full)
        EXPERIMENT_NAME="nerel-amtl-full"
        EXTRA_ARGS="--use_amtl --use_effective_number --use_negative_sampling --neg_pos_ratio 3.0 --use_per_class_thresholds"
        echo "=== Training with FULL PIPELINE (all techniques) ==="
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Usage: $0 [baseline|amtl|amtl-only|sampling-only|full]"
        exit 1
        ;;
esac

SAVE_PATH="${SAVE_BASE}/${EXPERIMENT_NAME}"

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Save path: $SAVE_PATH"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $NUM_EPOCHS"
echo "  Extra args: $EXTRA_ARGS"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please set DATA_DIR environment variable or create the directory."
    exit 1
fi

# Create save directory if it doesn't exist
mkdir -p "$SAVE_PATH"

# Run training
python3 run.py --do_train \
    --data_dir "$DATA_DIR" \
    --transformer_type "$TRANSFORMER_TYPE" \
    --model_name_or_path "$MODEL_NAME" \
    --display_name "$EXPERIMENT_NAME" \
    --save_path "$SAVE_PATH" \
    --train_file train.json \
    --dev_file dev.json \
    --test_file test.json \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_labels $NUM_LABELS \
    --lr_transformer $LR_TRANSFORMER \
    --max_grad_norm $MAX_GRAD_NORM \
    --evi_thresh $EVI_THRESH \
    --evi_lambda $EVI_LAMBDA \
    --warmup_ratio $WARMUP_RATIO \
    --num_train_epochs $NUM_EPOCHS \
    --seed $SEED \
    --num_class $NUM_CLASS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --max_sent_num $MAX_SENT_NUM \
    --eval_mode single \
    $EXTRA_ARGS

echo ""
echo "=== Training completed ==="
echo "Results saved to: $SAVE_PATH"
echo ""
echo "Check results:"
echo "  Overall metrics: $SAVE_PATH/scores.csv"
echo "  Per-relation metrics: $SAVE_PATH/scores_detailed.csv"
if [[ "$MODE" == "full" ]]; then
    echo "  Optimized thresholds: $SAVE_PATH/optimized_thresholds.json"
fi
echo ""
echo "To test the model:"
echo "  ./test_amtl.sh $EXPERIMENT_NAME"
