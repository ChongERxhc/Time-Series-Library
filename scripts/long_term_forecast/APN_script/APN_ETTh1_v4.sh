#!/bin/bash

# APN v4 - Following paper settings more closely
# Target: MSE < 0.4 and MAE < 0.4

export CUDA_VISIBLE_DEVICES=0

model_name=APN

# Following APN paper: higher learning rate, simpler patches
# Key: more patches with smaller coverage per patch

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id APN_ETTh1_96_96_v4 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --n_heads 4 \
  --d_ff 256 \
  --dropout 0.0 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 1 \
  --des 'APN_ETTh1_v4' \
  --itr 1 \
  --train_epochs 30 \
  --patience 10 \
  --learning_rate 0.01 \
  --batch_size 64 \
  --num_workers 4 \
  --apn_te_dim 16 \
  --apn_npatch 96 \
  --apn_nlayer 1 \
  --apn_attn_heads 4