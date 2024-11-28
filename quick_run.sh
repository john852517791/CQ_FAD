# tlmd=$2
# train_log_dir=$1
gpu=$1
model=$2
# loss=$5
dir=$3
q=$(echo "$model" | sed 's/\//_/g')
q=$(echo "$q" | sed 's/\.py//g')

model=$(echo "$model" | sed 's/\//./g')
model=$(echo "$model" | sed 's/\.py//g')



if [ -z "$loss" ] || [ "$loss" = "-" ]; then
    loss="CE"
fi

# CUDA_VISIBLE_DEVICES=${gpu} 
line="
nohup 
python main_eer.py 
--seed 123
--gpuid ${gpu} 
--tl_model models.a_tl_models.tl_model_trueRec
--module_model ${model}
--tl_data_module utils.loadData.asvspoof_data_toRGB_norm
--batch_size 8
--epochs 30
--no_best_epochs 5
--savedir z_train/colour16_head4_0.01/${dir}/${q} 
--optim adam
--optim_lr 0.00001
--weight_decay 0.01
--loss WCE
--scheduler none
--num_warmup_steps 5
--step_size 5
--gamma 0.5
--num_warmup_steps 10
--truncate 65600

--colour_num_colours 16
--colour_num_heads 4
--colour_out_channels 256
--colour_temperature 0.01

> a_nohup/${q}${gpu}.log
&
"

# --weight_decay 0.0001
# --usingDA
echo ${line}
eval ${line}