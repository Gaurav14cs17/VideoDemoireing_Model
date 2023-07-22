import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
'''
DeformConv2d : https://arxiv.org/pdf/1811.11168.pdf
'''

wn = lambda x: torch.nn.utils.weight_norm(x)


class Up_sample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Up_sample, self).__init__()
        self.upsample_layer = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample_layer(x)
        return x


class Stage_3_Module(nn.Module):
    def __init__(self, n_feats, groups=8):
        super(Stage_3_Module, self).__init__()
        self.conv_layer_1 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                      bias=True)
        self.conv_layer_2 = nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.L3_dcnpack = DeformConv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1,
                                         dilation=1, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, y):
        offset = torch.cat([x, y], dim=1)
        offset = self.lrelu(self.conv_layer_1(offset))
        offset = self.lrelu(self.conv_layer_2(offset))
        output_fea = self.lrelu(self.L3_dcnpack(x, offset))
        return offset, output_fea


class Stage_2_Module(nn.Module):
    def __init__(self, n_feats, groups=8):
        super(Stage_2_Module, self).__init__()
        self.L2_offset_conv1 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                         bias=True)
        self.L2_offset_conv2 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                         bias=True)
        self.L2_offset_conv3 = nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True)

        self.L2_dcnpack = DeformConv2d(n_feats, n_feats, 3, stride=1, padding=1, dilation=1, groups=groups)
        self.L2_fea_conv = nn.Conv2d(n_feats * 2, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.up_scale = Up_sample(scale_factor=2)

    def stage_forward(self, x, y, prev_offset):
        offset = torch.cat([x, y], dim=1)
        offset = self.lrelu(self.L2_offset_conv1(offset))
        offset = self.lrelu(self.L2_offset_conv2(torch.cat([offset, prev_offset * 2], dim=1)))
        offset = self.lrelu(self.L2_offset_conv3(offset))
        output_fea = self.L2_dcnpack(x, offset)
        return offset, output_fea

    def forward(self, x, y, prev_offset, prev_stage_fea):
        prev_offset = self.up_scale(prev_offset)
        offset, output_fea = self.stage_forward(x, y, prev_offset)
        prev_stage_fea = self.up_scale(prev_stage_fea)
        output_fea = self.lrelu(self.L2_fea_conv(torch.cat([output_fea, prev_stage_fea], dim=1)))
        return offset, output_fea


class Stage_1_Module(nn.Module):
    def __init__(self, n_feats, groups=8):
        super(Stage_1_Module, self).__init__()
        self.L1_offset_conv1 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                         bias=True)
        self.L1_offset_conv2 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                         bias=True)
        self.L1_offset_conv3 = nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                         bias=True)
        self.L1_dcnpack = DeformConv2d(n_feats, n_feats, 3, stride=1, padding=1, dilation=1, groups=groups)
        self.L1_fea_conv = nn.Conv2d(n_feats * 2, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.up_scale = Up_sample(scale_factor=2)

    def stage_forward(self, x, y, prev_offset):
        offset = torch.cat([x, y], dim=1)
        offset = self.lrelu(self.L1_offset_conv1(offset))
        offset = self.lrelu(self.L1_offset_conv2(torch.cat([offset, prev_offset * 2], dim=1)))
        offset = self.lrelu(self.L1_offset_conv3(offset))
        output_fea = self.L1_dcnpack(x, offset)
        return offset, output_fea

    def forward(self, x, y, prev_offset, prev_stage_fea):
        prev_offset = self.up_scale(prev_offset)
        offset, output_fea = self.stage_forward(x, y, prev_offset)

        prev_stage_fea = self.up_scale(prev_stage_fea)
        output_fea = self.lrelu(self.L1_fea_conv(torch.cat([output_fea, prev_stage_fea], dim=1)))
        return offset, output_fea


class PcdAlign(nn.Module):
    def __init__(self, n_feats=18, groups=3, wn=None):
        super(PcdAlign, self).__init__()
        self.stage_3 = Stage_3_Module(n_feats , groups)
        self.stage_2 = Stage_2_Module(n_feats , groups)
        self.stage_1 = Stage_1_Module(n_feats , groups)
        self.cas_offset_conv1 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True)
        self.cas_offset_conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True)
        self.cas_dcnpack = DeformConv2d(n_feats, n_feats, 3, stride=1, padding=1, dilation=1, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):

        offset_3, output_L_fea_3 = self.stage_3(nbr_fea_l[2], ref_fea_l[2])
        offset_2, output_L_fea_2 = self.stage_2(nbr_fea_l[1], ref_fea_l[1] , offset_3, output_L_fea_3)
        offset_1, output_L_fea_1 = self.stage_1(nbr_fea_l[0], ref_fea_l[0] , offset_2, output_L_fea_2)

        # Cascading
        offset = torch.cat([output_L_fea_1, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(output_L_fea_1, offset))
        return L1_fea




if __name__ == '__main__':
    new_model = PcdAlign()
    in_image = [torch.randn(3,18 ,16 , 16) , torch.randn(3, 18 , 8 ,8) , torch.randn(3, 18 , 4 , 4)]
    gt_image = [torch.randn(3,18 ,16 , 16) , torch.randn(3, 18 , 8 ,8) , torch.randn(3, 18 , 4 , 4)]
    new_output = new_model(in_image,gt_image)


    # groups =  1
    # model_obj = DCN(64, 64, 7, stride=1, padding=1, dilation=1, groups=groups)
    # image =  torch.randn( 3  , 64 , 32 , 32)
    # offset = torch.randn(3 ,  98 , 32 , 32)
    # print(model_obj(image, offset).shape)