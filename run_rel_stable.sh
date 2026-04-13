#!/bin/bash

# STABLE CONFIGURATION (No AMTL)
# Proven to achieve F1: 68.00% on NEREL document-level
# No training collapse, stable throughout 160 epochs

python3 run.py --do_train \
    --data_dir $HOME/data/relations/NEREL/naa-docred-split \
    --transformer_type roberta \
    --model_name_or_path ai-forever/ruRoberta-large \
    --display_name nerel-ruroberta-stable \
    --save_path ./logs/nerel-ruroberta-stable \
    --train_file train.json \
    --dev_file dev.json \
    --test_file test.json \
    --train_batch_size 1 \
    --test_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_labels 2 \
    --lr_transformer 3e-5 \
    --max_grad_norm 1.0 \
    --evi_thresh 0.2 \
    --evi_lambda 0.05 \
    --warmup_ratio 0.06 \
    --num_train_epochs 160 \
    --seed 7777 \
    --num_class 48 \
    --max_seq_length 1460 \
    --max_sent_num 101 \
    --eval_mode single
    # ✅ NO AMTL - uses default ATLoss (stable)
    # ✅ NO effective number weighting
    # ✅ NO negative sampling
    # ✅ NO per-class thresholds
