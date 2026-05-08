#!/bin/bash

# DREEAM training for BioNNE-R English
# Backbone: roberta-large (English-specific large model)
# num_class = 13 (12 BioNNE-R semantic relations after stripping ABBR/ALT_NAME + 1 Na)
# Verify num_class against your existing Russian rel2id.json before running.

python3 run.py --do_train \
    --data_dir $HOME/data/relations/BioNNE-R-docred/en \
    --transformer_type roberta \
    --model_name_or_path roberta-large \
    --display_name bionne-r-en-roberta-large \
    --save_path ./logs/bionne-r-en-roberta-large \
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
    --num_train_epochs 60 \
    --seed 7777 \
    --num_class 13 \
    --max_seq_length 1460 \
    --max_sent_num 101 \
    --eval_mode single
