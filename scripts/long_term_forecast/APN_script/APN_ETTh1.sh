#!/bin/bash

# APN (Adaptive Patching Network) for Multivariate Time Series Forecasting
# Paper: Rethinking Irregular Multivariate Time Series Forecasting: A Simple yet Effective Baseline (AAAI 2026)

# ETTh1 Dataset Configuration
# - 7 variables (features)
# - Hourly data
# - seq_len: 96, pred_len: 96 (following standard TSLib setting)

export CUDA_VISIBLE_DEVICES=0

model_name=APN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id APN_ETTh1_96_96 \
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
  --n_heads 8 \
  --d_ff 128 \
  --dropout 0.1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --des 'APN_ETTh1' \
  --itr 1 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate 0.001 \
  --batch_size 32 \
  --num_workers 4 \
  --apn_te_dim 16 \
  --apn_npatch 12 \
  --apn_nlayer 2 \
  --apn_attn_heads 8