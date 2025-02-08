

# mode 
d_model=256
n_heads=$((d_model/64))
d_ff=$((d_model*4))
e_layers=1
seq_len=512
pred_len=96
echo "parameters: d_model:${d_model},n_heads:${n_heads},d_ff:${d_ff},e_layers:${e_layers}"

num_gpu=$(python -c "import torch; print(torch.cuda.device_count())")

use_ddp=0
lradj=cosine
train_epochs=10
batch_size=16
learning_rate=5e-5
num_workers=3
cache_train_data=False
multi_data_schema=./configs/multi_data_schema_data.csv
model_prefix=tiny
echo $model_prefix

# ##################################### test zero-shot #####################################
is_training=0
is_test_zero=1
use_multi_gpu_gbl=False
use_ddp=0

# test on ETTH1
use_multi_gpu=$use_multi_gpu_gbl
batch_size=$batch_size 
zero_data=ETTh1
zero_root_path=./datasets/
zero_data_path=ETTh1.csv
python run.py \
    --is_training $is_training \
    --is_test_zero $is_test_zero \
    --model_prefix $model_prefix \
    --zero_data $zero_data \
    --zero_root_path  $zero_root_path \
    --zero_data_path $zero_data_path \
    --lradj $lradj \
    --train_epochs $train_epochs \
    --batch_size $batch_size \
    --e_layers $e_layers \
    --use_multi_gpu $use_multi_gpu \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --d_model $d_model \
    --n_heads $n_heads \
    --d_ff $d_ff \
    --use_ddp $use_ddp 

