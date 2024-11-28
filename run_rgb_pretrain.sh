train_log_dir=$1
gpu=$2
# loss=$5


model=$(echo "$model" | sed 's/\//./g')
model=$(echo "$model" | sed 's/\.py//g')


if [ -z "$loss" ] || [ "$loss" = "-" ]; then
    loss="CE"
fi

# CUDA_VISIBLE_DEVICES=${gpu} 
line="
nohup 
python main_batchloss.py 
--seed 1234
--gpuid ${gpu} 
--tl_model models.tl_model_pretrain
--module_model models.rgb_pretrain.Colour_Quantisation_1
--tl_data_module utils.loadData.asvspoof_data_rgb_vctk_norm
--batch_size 128
--epochs 5
--no_best_epochs 10
--savedir ${train_log_dir} 
--optim adamw
--optim_lr 0.001
--weight_decay 0.01
--loss mse
--scheduler cosWarmup
--num_warmup_steps 3

--colour_num_colours 16
--colour_num_heads 4
--colour_out_channels 256
--colour_temperature 0.001

> a_nohup/log${gpu}.log
&
"

# --step_size 1
# --weight_decay 0.0001
# --truncate 64000
# --usingDA
echo ${line}
eval ${line}