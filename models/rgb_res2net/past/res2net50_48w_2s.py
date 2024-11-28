from torch import nn
import torch 
import sys
sys.path.append('./')
from models.rgb_pretrain.Colour_Quantisation_1 import Model as pretrainedone
# from models.rgb_pretrain.aasist import AASIST
from models.rgb_res2net.model import res2net50_48w_2s

class Model(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.pretrained = pretrainedone(args)
        self.classifier = res2net50_48w_2s(False)
    
    def forward(self,x,train=False):
        transform_img, sec = self.pretrained(x,train)
        output = self.classifier(transform_img)
        
        return output,transform_img

    
if __name__ == "__main__":

    from utils.arg_parse import f_args_parsed
    args = f_args_parsed()
    args.colour_num_colours= 16
    args.colour_num_heads=4
    args.colour_out_channels= 256
    args.colour_temperature= 0.001
    device = torch.device("cuda:5")
    # img = torch.randn((8, 201,128)).to(device)
    img = torch.randn((8,3,256,256)).to(device)
    model = Model(args).to(device)
    output = model(img)
    
    print(output[0].shape)
    print(output[1].shape)   
    print(sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)) 
    