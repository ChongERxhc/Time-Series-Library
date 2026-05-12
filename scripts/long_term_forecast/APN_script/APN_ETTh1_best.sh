#!/bin/bash

# APN Best - High capacity model with longer training
# Target: MSE < 0.4 and MAE < 0.4

export CUDA_VISIBLE_DEVICES=0

model_name=APN

# High capacity with more patches, longer training

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id APN_ETTh1_best \
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
  --d_ff 256 \
  --dropout 0.1 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 1 \
  --des 'APN_best' \
  --itr 1 \
  --train_epochs 100 \
  --patience 20 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --num_workers 4 \
  --apn_te_dim 16 \
  --apn_npatch 32 \
  --apn_nlayer 3 \
  --apn_attn_heads 8