# tlmd=$2
# train_log_dir=$1
gpu=$1
model=$2
# loss=$5

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
--seed 1234
--gpuid ${gpu} 
--tl_model models.a_tl_models.tl_model_allRec
--module_model ${model}
--tl_data_module utils.loadData.asvspoof_data_toRGB
--batch_size 8
--epochs 100
--no_best_epochs 10
--savedir z_train/allRec/${q} 
--optim adam
--optim_lr 0.00001
--weight_decay 0.01
--loss WCE
--scheduler step
--step_size 10
--gamma 0.5
--num_warmup_steps 10
--truncate 65600
> a_nohup/${q}${gpu}.log
&
"

# --weight_decay 0.0001
# --usingDA
echo ${line}
eval ${line}