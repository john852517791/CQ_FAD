# --module-model models.pure.stable_aasist
ckpt=$1
gpu=$2


# --module-model models.rawformer.stable_base
line=" 
nohup python main_eer.py --inference 
--trained_model ${ckpt}
--batch_size 128
--gpuid ${gpu}

--colour_num_colours 16
--colour_num_heads 4
--colour_out_channels 256
--colour_temperature 0.01
--truncate 65600

> ${ckpt}/z_infer.log
&"

# --tl_model models.a_tl_models.tl_model_trueRec
# --module_model models.rgb_resnet.resnet18
# --tl_data_module utils.loadData.asvspoof_data_toRGB

# --colour_num_colours 2
# --colour_num_heads 4
# --colour_out_channels 256
# --colour_temperature 0.01



echo ${line}
eval ${line}