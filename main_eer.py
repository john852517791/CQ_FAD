import lightning as L
from utils.arg_parse import f_args_parsed,set_random_seed
import importlib
import os,yaml
from models import tl_model
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint,LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers
from typing import List,Tuple
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# arguments initialization
args = f_args_parsed()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

# train set, dev set
if True:
    # ⭐train 
    if not args.inference:
        data_util = importlib.import_module(args.tl_data_module)
        tl_md = importlib.import_module(args.tl_model)
        asvspoof_dm = data_util.asvspoof_dataModule(args=args)
        # import model.py
        prj_model = importlib.import_module(args.module_model)
        
        # random seed initialization and gpu seed 
        set_random_seed(args.seed, args)
        # model 
        model = prj_model.Model(args)

        # init model, including loss func and optim 
        customed_model_wrapper = tl_md.base_model(
            model=model,
            args=args
            )

        # config logdir
        tb_logger = pl_loggers.TensorBoardLogger(args.savedir,name="")
        
        # model initialization
        trainer = L.Trainer(
            max_epochs=args.epochs,
            strategy='ddp_find_unused_parameters_true',
            log_every_n_steps = 1,
            callbacks=[
                # dev损失无下降就提前停止
                EarlyStopping('dev_eer',patience=args.no_best_epochs,mode="min",verbose=True,log_rank_zero_only=True),
                # 模型按照最低val_loss来保存
                ModelCheckpoint(monitor='dev_eer',
                                save_top_k=1,
                                save_weights_only=True,mode="min",filename='best_model-{epoch:02d}-{dev_eer:.6f}'),
                LearningRateMonitor(logging_interval='epoch',log_weight_decay=True),
                ],
            check_val_every_n_epoch=1,
            logger=tb_logger,
            enable_progress_bar=False
            )
        trainer.fit(
            model=customed_model_wrapper, 
            datamodule=asvspoof_dm
            )
        
        trainer.test(
            model=customed_model_wrapper,
            datamodule=asvspoof_dm,
            verbose=False
            )
   
    else:
        
        # args.testset = "DF21"
        
        checkpointpath=args.trained_model
        # checkpointpath=trainer.log_dir
        args.savedir = checkpointpath
        
        # gain model
        ymlconf = os.path.join(checkpointpath,"hparams.yaml")
        with open(ymlconf,"r") as f_yaml:
            parser1 = yaml.safe_load(f_yaml)
        args.module_model = parser1["module_model"]
        args.tl_model = parser1["tl_model"]
        args.tl_data_module = parser1["tl_data_module"]
        
        args.colour_num_colours = parser1["colour_num_colours"]
        args.colour_num_heads = parser1["colour_num_heads"]
        args.colour_out_channels = parser1["colour_out_channels"]
        args.colour_temperature = parser1["colour_temperature"]
        infer_m = importlib.import_module(args.module_model)
        tl_md = importlib.import_module(args.tl_model)
        infer_dm = importlib.import_module(args.tl_data_module)
            
        infer_model = infer_m.Model(args)
        asvspoof_dm = infer_dm.asvspoof_dataModule(args=args)
        
        # print(args.savedir)
        ckpt_files = [file for file in os.listdir(checkpointpath+"/checkpoints/") if file.endswith(".ckpt")]
        # customed_model=model_wrapper.base_model(model=model)
        customed_model=tl_md.base_model.load_from_checkpoint(
            checkpoint_path=os.path.join(f"{checkpointpath}/checkpoints/",ckpt_files[0]),
            model=infer_model,
            args = args,
            strict=False)
        inferer = L.Trainer(logger=pl_loggers.TensorBoardLogger(args.savedir,name=""))
        
        # la19
        inferer.test(
            model=customed_model,
            datamodule=asvspoof_dm
            )
         # la21
        # inferer.predict(
        #     model=customed_model,
        #     datamodule=asvspoof_dm
        #     )
        # # df21
        # inferer.model.args.testset = "ITW"
        # asvspoof_dm = data_util.asvspoof_dataModule(args=args)
        # inferer.predict(
        #     model=customed_model,
        #     datamodule=asvspoof_dm
        #     )
        
        # # ITW
        # inferer.model.args.testset = "DF21"
        # asvspoof_dm = data_util.asvspoof_dataModule(args=args)
        # inferer.predict(
        #     model=customed_model,
        #     datamodule=asvspoof_dm
        #     )

        
        
        # inferfolder = os.path.join(checkpointpath,"infer")
        # if not os.path.exists(inferfolder):
        #     os.makedirs(inferfolder)
        # # 遍历文件夹A中的所有文件
        # folder_a = os.path.join(checkpointpath,"version_0")
        # for filename in os.listdir(folder_a):
        #     if filename.endswith('.log'):  # 检查文件名是否以'.log'结尾
        #         # 构造原始文件和目标文件的完整路径
        #         original_path = os.path.join(folder_a, filename)
        #         destination_path = os.path.join(inferfolder, filename)
                
        #         # 移动文件
        #         shutil.move(original_path, destination_path)

        # # 删除文件夹A及其所有内容
        # shutil.rmtree(folder_a)
# print(args)