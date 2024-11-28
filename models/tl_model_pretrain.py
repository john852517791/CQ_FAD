from typing import Any, List, Union
import lightning as L
# from lightning.pytorch.utilities.types import LRSchedulerPLType
import torch
import logging,os
from utils.wrapper import loss_wrapper, optim_wrapper,schedule_wrapper   
from utils.tools import cul_eer 
import numpy as np
import math
from models.rgb_pretrain.convert_RGB_HSV import RGB_HSV

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
        self.logging_test = None
        self.logging_predict = None
        
        self.convertor = RGB_HSV()
        self.num_colors = args.colour_num_colours
        self.alpha = args.colour_alpha
        self.beta = args.colour_beta
        self.gamma = args.colour_gamma
    
    def HSVDistance(self, hsv_1, hsv_2):
        H_1, S_1, V_1 = hsv_1.split(1, dim=1)
        H_2, S_2, V_2 = hsv_2.split(1, dim=1)
        H_1 = H_1 * 360
        H_2 = H_2 * 360
        R = np.sqrt(2)
        angle = 30
        h = R * math.cos(angle / 180 * math.pi)
        r = R * math.sin(angle / 180 * math.pi)
        x1 = r * V_1 * S_1 * torch.cos(H_1 / 180 * torch.pi)
        y1 = r * V_1 * S_1 * torch.sin(H_1 / 180 * torch.pi)
        z1 = h * (1 - V_1)
        x2 = r * V_2 * S_2 * torch.cos(H_2 / 180 * torch.pi)
        y2 = r * V_2 * S_2 * torch.sin(H_2 / 180 * torch.pi)
        z2 = h * (1 - V_2)
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        return dx * dx + dy * dy + dz * dz

        
    def forward(self,x,training=False):
        return self.model(x,training)
    
    def training_step(self, batch, batch_idx):
        
        # training_step defines the train loop.
        # model output
        B = batch.shape[0]
        transformed_img,probability_map = self.forward(batch,training=True)
        prob_max, _ = torch.max(probability_map.view([B, self.num_colors, -1]), dim=2)
        avg_max = torch.mean(prob_max)
        # reconsturction_loss
        rgb_color_palette = (
            (batch.unsqueeze(2) * probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True)
            / (probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True)+ 1e-8)
            )
        hsv = self.convertor.rgb_to_hsv(batch)
        hsv_color_palette = self.convertor.rgb_to_hsv(rgb_color_palette.view(B, 3, self.num_colors, 1)) \
            .unsqueeze(-1)
        hsv_color_contribution = (hsv.unsqueeze(2) * probability_map.unsqueeze(1))
        hsv_color_var = (self.HSVDistance(hsv_color_contribution, hsv_color_palette) *
                            probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                                probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
        # Shannon_entropy
        probability_map_mean = torch.mean(probability_map.view([B, self.num_colors, -1]), dim=2)
        Shannon_entropy = -probability_map_mean * torch.log2(torch.tensor([1e-8], device='cuda') + probability_map_mean)
        Shannon_entropy = torch.sum(Shannon_entropy, dim=1)
        Shannon_entropy = torch.mean(Shannon_entropy)
        
         
        colorvar = self.alpha * (hsv_color_var.mean())
        beta_num_color = self.beta * np.log2(self.num_colors) * (1 - avg_max)
        recon = self.gamma * self.loss_criterion(batch,transformed_img).mean()

        batch_loss = colorvar + beta_num_color + recon
        
        # self.logger.experiment.add_image()
        self.log_dict({
            "a_colorvar": colorvar,
            "a_beta_num_color": beta_num_color,
            "a_recon": recon,
            "batch_loss": batch_loss,
            },on_step=True, 
                on_epoch=True,prog_bar=True, logger=True,
                # prevent from saving wrong ckp based on the eval_loss from different gpus
                sync_dist=True, 
                )
        return batch_loss
        
    def validation_step(self,batch):
        transformed_img,probability_map = self.forward(batch)
        tensorboard = self.logger.experiment
        for i in range(batch.shape[0]):
            tensorboard.add_image(
                f'recon_{i}', transformed_img[i], self.global_step
            )
            tensorboard.add_image(
                f'ori_{i}', batch[i], self.global_step
            )
        
         
 
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