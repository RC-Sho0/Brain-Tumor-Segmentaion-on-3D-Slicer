from monai.networks.nets import DynUNet, SegResNet, VNet, SwinUNETR
from model.v2.DynUnet_DDA import DynUNet_DDA
import torch
import torch.nn as nn
from model.st.reg import Out, ConvBNReLU
# from reg import Out, ConvBNReLU


def get_model(name, att = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == "dynunet":
        model = DynUNet(
                    spatial_dims=3,
                    in_channels=4,
                    out_channels=3,
                    filters = [32, 64, 128, 256, 320, 320],
                    kernel_size= [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                    strides= [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                    upsample_kernel_size=[ [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                    norm_name="instance",
                    deep_supervision=False,
                    deep_supr_num=1,).to(device)
        return model

    elif name == "segresnet":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,).to(device)
        return model
    elif name == 'dynunet_dda':
        print(att)
        model = DynUNet_DDA(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            filters = [32, 64, 128, 256, 320, 320],
            kernel_size= [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides= [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[ [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            norm_name="instance",
            deep_supervision=False,
            deep_supr_num=1,
            attention = att).to(device)
        return model
    elif name == "swinunet":
        model = SwinUNETR(
            img_size=(128,128,128),
            in_channels=4,
            out_channels=3,
            drop_rate=0.3,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,).to(device)
        return model

    elif name == "vnet":
        model = VNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.5,
            ).to(device)
        return model
    else:
        model = None


def get_down(name, trained, cate=False):
    ## Init model
    model = get_model(name)
    ## Load n Frozen model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(trained ,map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    for param in model.parameters():
        param.requires_grad = False
    
    net = Out(320, cate).to(device)
    ## Get encoder part:
    if name == 'dynunet':
        down = torch.nn.Sequential(model.input_block, *model.downsamples[:],model.bottleneck)
        net = torch.nn.Sequential(ConvBNReLU(320,320,3,1),ConvBNReLU(320,320,3,1), net).to(device)
        return net, down
    elif name == 'segresnet':
        down = torch.nn.Sequential(model.act_mod, model.convInit, *model.down_layers)
        net = torch.nn.Sequential(ConvBNReLU(128,256,3,2),ConvBNReLU(256,320,3,2), net).to(device)
        return net, down
        # torch.nn.Conv3d(512,320,1,1)

if __name__ == "__main__":
    ## params
    name = "segresnet"
    trained = "temp/trained_segresnet.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## getmodel
    x = torch.zeros((2,4,128,128,128)).to(device)
    reg, encode = get_down(name, trained, cate=True)
    model = nn.Sequential()
    model.add_module("encode", encode)
    model.add_module("reg", reg)
    y = model(x)
    print(y)
