import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv3d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat
        
class Ex(nn.Module):
    def __init__(self, in_c):
        super(Ex, self).__init__()
        self.left = nn.Sequential(
            ConvBNReLU(in_c, in_c, 1, stride=1, padding=0),
            ConvBNReLU(in_c, in_c, 3, stride=2),
        )
        self.right = nn.MaxPool3d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = nn.Conv3d(in_c * 2, in_c * 2 , 1, stride=1)

    def forward(self, x):
        feat_left = self.left(x)
        feat_right = self.right(x)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        # print(f"shape {feat.shape}")
        return feat

class Reg(nn.Module):
    def __init__(self, cate=False):
        super(Reg, self).__init__()
        
        self.flat = nn.Flatten()
        self.dens = nn.Sequential(
            nn.Linear(5120, 1208),
            nn.Dropout(0.1),
            nn.ReLU(True),)
        self.reg = nn.Linear(1208, 1)
        self.categor = nn.Linear(1208, 3)
        self.cate = cate

    def forward(self, x):
        
        x = self.flat(x)
        x = self.dens(x)
        reg = self.reg(x)
        if self.cate == True:
            clas = self.categor(x)
            return reg, clas
        else:
            return reg

class Out(nn.Module):
    def __init__(self, in_c, cate=False):
        super(Out, self).__init__()

        self.ex = Ex(in_c)
        self.reg = Reg(cate)
        self.cate = cate
    
    def forward(self, x):
        out = self.ex(x)
        if self.cate:
            out1, out2 = self.reg(out)
            return out1, out2
        else:
            out = self.reg(out)
            return out

if __name__ == "__main__":
    import numpy as np
    x = torch.zeros((2,320,4,4,4))
    model = Out(320, True)
    y = model(x)
    print(y)