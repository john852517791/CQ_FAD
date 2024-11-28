from typing import Any, List, Union
import lightning as L
# from lightning.pytorch.utilities.types import LRSchedulerPLType
import torch
import logging,os
from utils.wrapper import loss_wrapper, optim_wrapper,schedule_wrapper   
from utils.tools import cul_eer 



class base_model(L.LightningModule):
    def __init__(self, 
                 model,
                 args,
                 ) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.save_hyperparameters(self.args)
        
        self.model_optimizer = optim_wrapper.optimizer_wrap(self.args, self.model).get_optim()
        self.LRScheduler = schedule_wrapper.scheduler_wrap(self.model_optimizer,self.args).get_scheduler()
        # for loss
        self.args.model = model
        self.args.samloss_optim = self.model_optimizer
        self.loss_criterion,self.loss_optimizer,self.minimizor = loss_wrapper.loss_wrap(self.args).get_loss()
        
        self.mseloss = torch.nn.MSELoss(reduction="none")
        self.logging_test = None
        self.logging_predict = None
        
    def forward(self,x,train=False):
        return self.model(x,train)
    
    def training_step(self, batch, batch_idx):
        
        # training_step defines the train loop.
        # it is independent of forward
        data_in, data_label,name = batch
        # model output
        data_predict,transform_img = self.forward(data_in,train=True)
        ce_loss = self.loss_criterion(data_predict, data_label)
        
        mask = (data_label == 1).float()
        
        recon_loss = self.mseloss(transform_img,data_in)
        recon_loss_all = []
        for i in range(recon_loss.shape[0]):
            recon_loss_all.append(recon_loss[i].mean().item())
        # print(recon_loss_all)
        recon_loss_all = torch.tensor(recon_loss_all).to(mask.device)
        masked_true_recon_loss = recon_loss_all * mask
        
        batch_loss = (ce_loss + masked_true_recon_loss * self.args.recon_weight).mean()
        
        self.log_dict({
            "a_ce_loss": ce_loss.mean(),
            "a_recon_loss": masked_true_recon_loss.mean(),
            "loss": batch_loss,
            },on_step=True, 
                on_epoch=True,prog_bar=True, logger=True,
                # prevent from saving wrong ckp based on the eval_loss from different gpus
                sync_dist=True, 
                )
        return batch_loss
        
    def validation_step(self,batch):
        # training_step defines the train loop.
        # it is independent of forward
        data_in, dataname = batch
        # model output
        data_predict,_ = self.forward(data_in)
                
        data_predict = torch.nn.functional.softmax(data_predict,dim=1)
                
        # log the prediction for cul eer
        with open(os.path.join(self.logger.log_dir,"dev.log"), 'a') as file:
            for i in range(len(data_predict)):
                # file.write(f"{dataname[i]} - {data_label[i]} {str(data_predict.cpu().numpy()[i][1])}\n")
                file.write(f"{dataname[i]} {str(data_predict.cpu().numpy()[i][1])}\n")
        
        # batch_loss = self.loss_criterion(data_predict, data_label).mean()
        # # Logging to TensorBoard (if installed) by default
        # self.log("val_loss", batch_loss, batch_size=len(data_in),sync_dist=True)
    
    def on_validation_epoch_end(self) -> None:
        dev_eer = 0.
        dev_tdcf = 0.
        with open(os.path.join(self.logger.log_dir,"dev.log"), 'r') as file:
            lines = file.readlines()

        if len(lines) > 10000:
            dev_eer, dev_tdcf = cul_eer.eerandtdcf(
                os.path.join(self.logger.log_dir,"dev.log"),
                "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/datasets/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
                "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/datasets/asvspoof2019/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
            )
        with open(os.path.join(self.logger.log_dir,"dev.log"), 'w') as file:
            pass
        self.log_dict({
            "dev_eer": (dev_eer),
            "dev_tdcf": dev_tdcf,
            },on_step=False, 
                on_epoch=True,prog_bar=False, logger=True,
                # prevent from saving wrong ckp based on the eval_loss from different gpus
                sync_dist=True, 
                )
        
    def on_test_start(self):
        # logging.basicConfig(filename=os.path.join(self.logger.log_dir,f"infer_test.log"),level=logging.INFO,format="")
        self.logging_test = logging.getLogger("logging_test")
        self.logging_test.setLevel(logging.INFO)
        hdl=logging.FileHandler(os.path.join(self.logger.log_dir,f"infer_19.log"))
        hdl.setFormatter("")
        self.logging_test.addHandler(hdl)        
        
    def test_step(self, batch,) -> Any:
        data_in, data_info = batch[0],batch[1]
        data_predict,_ = self.forward(data_in,False)
        
        data_predict = torch.nn.functional.softmax(data_predict,dim=1)
        
        for i in range(len(data_info)):
            self.logging_test.info(f"{data_info[i]} {batch[2][i]} {batch[3][i]} {str(data_predict.cpu().numpy()[i][0])} {str(data_predict.cpu().numpy()[i][1])}")
        # return data_info[0],data_predict.cpu().numpy()
        return {'loss': 0, 'y_pred': data_predict}
    
    def on_predict_start(self):
        # logging.basicConfig(filename=os.path.join(self.args.savedir,f"infer_predict.log"),level=logging.INFO,format="")
        self.logging_predict = logging.getLogger(f"logging_predict_{self.args.testset}")
        self.logging_predict.setLevel(logging.INFO)
        hdlx = logging.FileHandler(os.path.join(self.logger.log_dir,f"infer_{self.args.testset}.log"))
        hdlx.setFormatter("")
        self.logging_predict.addHandler(hdlx)
    
    def predict_step(self, batch, batch_idx):
        data_in, data_info = batch
        data_predict,_,_ = self.forward(data_in)
        
        data_predict = torch.nn.functional.softmax(data_predict,dim=1)
        
        # self.logging_predict.info(f"{data_info[0]} {str(data_predict.cpu().numpy()[0][1])} {str(data_predict.cpu().numpy()[0][0])}")
        for i in range(len(data_info)):
            self.logging_predict.info(f"{data_info[i]} {str(data_predict.cpu().numpy()[i][1])}")
        # return data_info[0],data_predict.cpu().numpy()
        return 

    def configure_optimizers(self):
        configure = None
        if self.LRScheduler is not None:
            configure = {
                "optimizer":self.model_optimizer,
                'lr_scheduler': self.LRScheduler, 
                'monitor': 'val_loss'
                }
        else:
            configure = {
                "optimizer":self.model_optimizer,
                }
            
        return configure