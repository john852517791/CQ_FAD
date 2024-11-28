from torch import optim
from transformers import get_cosine_schedule_with_warmup


class schdule_conf():
    def __init__(self):
        self.scheduler = "cosWarmup"
        self.epochs = 100
        # for cosWarmup
        self.num_warmup_steps = 5
        # self.num_training_steps = self.epochs - self.num_warmup_steps
        # for cosanneal 
        self.total_step = 1057 # (25380//24) * 100
        # for step
        self.step_size = 5
        self.gamma = 0.1


class scheduler_wrap():
    """ Wrapper over different types of learning rate Scheduler
    
    """
    def __init__(self, optimizer, args:schdule_conf): 
        self.optimizer = optimizer
        self.args = args
        
    def get_scheduler(self):
        
        # other config or none
        scheduler = None   
         
        if  self.args.scheduler == "cosWarmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer = self.optimizer, 
                num_warmup_steps=self.args.num_warmup_steps,          
                num_training_steps = self.args.epochs
            )
        elif self.args.scheduler == "cosAnneal":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=(25380//self.args.batch_size),
                # T_max=(25380//self.args.batch_size) * self.args.epochs,
                eta_min=0
                )
        elif self.args.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.args.step_size, 
                gamma=self.args.gamma
                )
        return scheduler