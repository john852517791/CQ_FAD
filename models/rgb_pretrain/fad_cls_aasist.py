from torch import nn
import torch 
import sys
sys.path.append('./')
from models.rgb_pretrain.Colour_Quantisation_1 import Model as pretrainedone
from models.rgb_pretrain.aasist import AASIST


class Model(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.classifier = AASIST()
    
    def forward(self,x,train=False):
        bs, chennal, temporal, spectral = x.shape
        x1 = x.reshape(bs,-1,spectral)
        hidenstate , output = self.classifier(x1)
        
        return output,x,hidenstate

    
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
    print(output[2].shape)   
    print(sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)) 
    