export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# Configurable dataset: ETTh1 or ETTh2
dataset=${1:-ETTh1}
data_path=${dataset}.csv

for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path $data_path \
    --model_id ${dataset}_96_${pred_len} \
    --model $model_name \
    --data $dataset \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1
done
