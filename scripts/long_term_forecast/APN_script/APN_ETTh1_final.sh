#!/bin/bash

# APN Final - Optimized for regular time series
# Using finer patches and stronger regularization

export CUDA_VISIBLE_DEVICES=0

model_name=APN

# Key insight: For regular TS, use more patches with smaller coverage
# This allows APN's adaptive boundaries to work better

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id APN_ETTh1_final \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 64 \
  --n_heads 4 \
  --d_ff 128 \
  --dropout 0.2 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --des 'APN_final' \
  --itr 1 \
  --train_epochs 50 \
  --patience 15 \
  --learning_rate 0.005 \
  --batch_size 32 \
  --num_workers 4 \
  --apn_te_dim 8 \
  --apn_npatch 48 \
  --apn_nlayer 2 \
  --apn_attn_heads 4