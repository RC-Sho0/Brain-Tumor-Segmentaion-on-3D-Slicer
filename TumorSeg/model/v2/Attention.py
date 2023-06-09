
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.v2.DualDomainNet import DualDomainNet
from monai.networks.blocks import Convolution, UpSample



class Attention(nn.Module):
    def __init__(self, out_c, in_c, skip= 0):
        super(Attention, self).__init__()
        self.attention = DualDomainNet(out_c, in_c, skip)
        
        
    def forward(self, x):
        size = x.size()[2:]
        y =  self.attention(x)
        y =  UpSample(spatial_dims=3, mode='nontrainable', size=size)(y)
        return y
    
    

if __name__ == '__main__':
    
    model = Attention(4,128)
    x = torch.zeros((1, 4, 128, 128, 128))
    y_ = model(x)
    print(y_.shape)
