#!/bin/bash

# APN Hyperparameter Tuning for ETTh1
# Target: MSE < 0.4 and MAE < 0.4

export CUDA_VISIBLE_DEVICES=0

model_name=APN

# Larger model with more patches
# Key changes: larger d_model, more patches, higher te_dim, lower lr, more epochs

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id APN_ETTh1_96_96_v2 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --n_heads 8 \
  --d_ff 512 \
  --dropout 0.05 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --des 'APN_ETTh1_v2' \
  --itr 1 \
  --train_epochs 30 \
  --patience 10 \
  --learning_rate 0.0005 \
  --batch_size 32 \
  --num_workers 4 \
  --apn_te_dim 32 \
  --apn_npatch 24 \
  --apn_nlayer 3 \
  --apn_attn_heads 8