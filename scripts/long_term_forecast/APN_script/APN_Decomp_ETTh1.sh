#!/bin/bash

# APN with Decomposition - Seasonal + Trend
# Target: MSE < 0.4 and MAE < 0.4

export CUDA_VISIBLE_DEVICES=0

model_name=APN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id APN_Decomp_ETTh1 \
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
  --moving_avg 25 \
  --des 'APN_Decomp' \
  --itr 1 \
  --train_epochs 30 \
  --patience 10 \
  --learning_rate 0.001 \
  --batch_size 32 \
  --num_workers 4 \
  --apn_te_dim 16 \
  --apn_npatch 12 \
  --apn_nlayer 2 \
  --apn_attn_heads 8