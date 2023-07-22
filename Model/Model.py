import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Model(nn.Module):
    def __init__(self, n_feats, k_size=3, bias=True):
        super(Conv_Model, self).__init__()
        self.conv_layer = nn.Conv2d(n_feats, 2 * n_feats, kernel_size=(k_size, k_size), stride=(2, 2), padding=(1, 1),
                                    bias=bias)
        self.conv_layer_1 = nn.Conv2d(2 * n_feats, n_feats, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                      bias=bias)

    def forward(self, x):
        x = self.conv_layer_1(self.conv_layer(x))
        return x


class Stage_1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Stage_1, self).__init__()
        self.con_layer = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1))
        self.con_block = Conv_Model(out_ch)

    def forward(self, x):
        cur_lv1 = self.con_layer(x)
        cur_lv2 = self.con_block(cur_lv1)
        cur_lv3 = self.con_block(cur_lv2)

        cur_lv1_1 = F.interpolate(cur_lv1, scale_factor=0.5, mode='bilinear', align_corners=False)
        cur_lv1_2 = F.interpolate(cur_lv1, scale_factor=0.25, mode='bilinear', align_corners=False)
        cur_feats = [cur_lv1, cur_lv1_1, cur_lv1_2]

        return cur_feats , [cur_lv1 , cur_lv2, cur_lv3]



class Stage_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Stage_2, self).__init__()
        self.con_layer = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1))
        self.pcdalign = PcdAlign.PcdAlign(nf=out_ch)
        self.con_block = Conv_Model(out_ch)

    def forward(self, x , cur_feats ):
        ref_lv1 = self.con_layer(x)

        ref_lv1_1 = F.interpolate(ref_lv1, scale_factor=0.5, mode='bilinear', align_corners=False)
        ref_lv1_2 = F.interpolate(ref_lv1, scale_factor=0.25, mode='bilinear', align_corners=False)
        ref_feats = [ref_lv1, ref_lv1_1, ref_lv1_2]

        T_lv1 = self.pcdalign(nbr_fea_l=ref_feats, ref_fea_l=cur_feats)

        T_lv2 = self.con_block(T_lv1)
        T_lv3 = self.con_block(T_lv2)
        return ref_feats , [T_lv1 , T_lv2, T_lv3]




class VDM_PCD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(VDM_PCD, self).__init__()
        self.use_shuffle = True
        if self.use_shuffle:
            in_channels = 12
        else:
            in_channels = 3

        self.stage_layer_1 = Stage_1(in_channels ,32)
        self.stage_layer_2 = Stage_2(in_channels ,32)

    def down_shuffle(self, x, r):
        b, c, h, w = x.size()
        out_channel = c * (r ** 2)
        out_h = h // r
        out_w = w // r
        x = x.view(b, c, out_h, r, out_w, r)
        out = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)
        return out

    def forward(self, cur=None, ref=None, label=None, blend=1):

        if self.use_shuffle:
            cur = self.down_shuffle(cur, 2)

        cur_feats, cur_lv1, cur_lv2, cur_lv3 = self.stage_layer(cur)

        # aligned features
        align_feats_lv1 = cur_lv1
        align_feats_lv2 = cur_lv2
        align_feats_lv3 = cur_lv3

        for i in range(self.args.NUM_AUX_FRAMES):
            ref_tmp = ref[:, (0 + 3 * i):(3 + 3 * i), :, :]
            if self.use_shuffle:
                ref_tmp = self.down_shuffle(ref_tmp, 2)

            ref_feats, T_lv1, T_lv2, T_lv3 = self.stage_layer_2(ref_tmp)
            align_feats_lv1 = torch.cat((align_feats_lv1, T_lv1), dim=1)
            align_feats_lv2 = torch.cat((align_feats_lv2, T_lv2), dim=1)
            align_feats_lv3 = torch.cat((align_feats_lv3, T_lv3), dim=1)











