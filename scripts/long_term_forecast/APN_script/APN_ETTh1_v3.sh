#!/bin/bash

# APN v3 - Different approach with fewer patches and different architecture
# Target: MSE < 0.4 and MAE < 0.4

export CUDA_VISIBLE_DEVICES=0

model_name=APN

# Key changes:
# - Fewer patches (8) with larger per-patch coverage
# - Larger d_model (256)
# - Higher capacity decoder
# - Label smoothing style approach with label_len usage

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id APN_ETTh1_96_96_v3 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --n_heads 8 \
  --d_ff 1024 \
  --dropout 0.1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --des 'APN_ETTh1_v3' \
  --itr 1 \
  --train_epochs 50 \
  --patience 15 \
  --learning_rate 0.0001 \
  --batch_size 16 \
  --num_workers 4 \
  --apn_te_dim 64 \
  --apn_npatch 8 \
  --apn_nlayer 4 \
  --apn_attn_heads 8