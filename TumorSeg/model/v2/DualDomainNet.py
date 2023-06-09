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


class ContentBranch(nn.Module):

    def __init__(self, in_c, skip):
        super(ContentBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(in_c, 32, 3, stride=2),
            ConvBNReLU(32, 32, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(32, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )
        
        self.branch = nn.Sequential(
            self.S1 if skip < 3 else nn.Identity(),
            self.S2 if skip < 2 else nn.Identity(),
            self.S3 if skip < 1 else nn.Identity(),
        )

    def forward(self, x):
        feat = self.branch(x)
        return feat



class StemBlock(nn.Module):

    def __init__(self, in_c):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(in_c, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool3d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 32, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat



class C2Block(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(C2Block, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv3d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm3d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm3d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class C1Block(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(C1Block, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv3d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm3d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv3d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm3d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm3d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm3d(in_chan),
                nn.Conv3d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm3d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat



class SpatialBranch(nn.Module):

    def __init__(self, in_c, skip):
        super(SpatialBranch, self).__init__()
        self.S1S2 = StemBlock(in_c)
        self.S3 = nn.Sequential(
            C1Block(32, 32),
            C2Block(32, 32),
        )
        self.S4 = nn.Sequential(
            C1Block(32, 64),
            C2Block(64, 64),
        )
        self.S5 = nn.Sequential(
            C1Block(64, 128),
            C2Block(128, 128),
            C2Block(128, 128),
            C2Block(128, 128),
        )
        
        self.branch = nn.Sequential(
            self.S1S2,
            self.S3 if skip < 3 else nn.Identity(),
            self.S4 if skip < 2 else nn.Identity(),
            self.S5 if skip < 1 else nn.Identity(),
        )

    def forward(self, x):
        feat = self.branch(x)
        return feat



class MergeLayer(nn.Module):

    def __init__(self, in_c, skip):
        super(MergeLayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv3d(
                in_c, 128, kernel_size=3, stride=1,
                padding=1, groups=in_c, bias=False),
            nn.BatchNorm3d(128),
            nn.Conv3d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv3d(
                in_c, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.AvgPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv3d(
                in_c, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm3d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv3d(
                in_c, 128, kernel_size=3, stride=1,
                padding=1, groups=in_c, bias=False),
            nn.BatchNorm3d(128),
            nn.Conv3d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True), 
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = F.interpolate(
            right1, size=dsize, mode='trilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(
            right, size=dsize, mode='trilinear', align_corners=True)
        out = self.conv(left + right)
        return out


class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, num_classes):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv3d(
                mid_chan, num_classes, kernel_size=1, stride=1,
                padding=0, bias=True)

    def forward(self, x, size=None):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        if not size is None:
            feat = F.interpolate(feat, size=size,
                mode='trilinear', align_corners=True)
        return feat

    
class DualDomainNet(nn.Module):

    def __init__(self, num_classes, in_c, skip = 0):
        super(DualDomainNet, self).__init__()
        m_in = [128, 64, 32, 16]
        self.detail = ContentBranch(in_c, skip)
        if skip < 3:
            self.segment = SpatialBranch(in_c, skip) 
            self.merge = MergeLayer(m_in[skip], skip)
        in_seg = 128 if skip < 3 else m_in[::-1][skip-1] 
        if (in_c > 128) & (skip == 3): 
            in_seg = in_c
        self.head = SegmentHead(in_seg, 1024, num_classes)
        self.skip = skip
        
    def forward(self, x):
        size = x.size()[2:]
        feat_d = self.detail(x)
        if self.skip < 3:
            feat_s = self.segment(x)
            feat_head = self.merge(feat_d, feat_s)
        else:
            feat_head = feat_d
        logits = self.head(feat_head, size)
#         print(logits.shape)
        return logits

if __name__ == '__main__':
    model = DualDomainNet(num_classes = 32,in_c = 32,skip=0)
    x = torch.zeros(1,32,128,128,128)
    print(model(x).shape)
