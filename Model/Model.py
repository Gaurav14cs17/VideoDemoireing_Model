import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import MainNet, PcdAlign
import torchvision ,os


def get_multi_level(ref_lv1):
    ref_lv1_1 = F.interpolate(ref_lv1, scale_factor=0.5, mode='bilinear', align_corners=False)
    ref_lv1_2 = F.interpolate(ref_lv1, scale_factor=0.25, mode='bilinear', align_corners=False)
    ref_feats = [ref_lv1, ref_lv1_1, ref_lv1_2]
    return ref_feats


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
    def __init__(self, n_feat):
        super(Stage_1, self).__init__()
        self.con_block = Conv_Model(n_feat)

    def forward(self, cur_lv1):
        cur_lv2 = self.con_block(cur_lv1)
        cur_lv3 = self.con_block(cur_lv2)
        return cur_lv1, cur_lv2, cur_lv3


class Stage_2(nn.Module):
    def __init__(self, n_feat):
        super(Stage_2, self).__init__()
        self.pcdalign = PcdAlign.PcdAlign(nf=n_feat)
        self.con_block = Conv_Model(n_feat)

    def forward(self, ref_feats, cur_feats):
        T_lv1 = self.pcdalign(nbr_fea_l=ref_feats, ref_fea_l=cur_feats)
        T_lv2 = self.con_block(T_lv1)
        T_lv3 = self.con_block(T_lv2)
        return T_lv1, T_lv2, T_lv3


class VDM_PCD(nn.Module):
    def __init__(self, args, pretrain=False, freeze=False):
        super(VDM_PCD, self).__init__()
        self.args = args
        self.use_shuffle = args.use_shuffle
        self.backbone = args.backbone
        self.num_res_blocks = list(map(int, args.num_res_blocks.split('+')))

        # Demoireing backbone
        if self.backbone == 'vdm_pcd_v1':
            self.MainNet_V1 = MainNet.MainNet_V1(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats,
                                                 res_scale=args.res_scale, use_shuffle=args.use_shuffle)

        # # If true, load pretrained weights trained on other demoire dataset.
        if pretrain:
            self.load_weight(freeze)


        # RGB image with 3 channels
        self.use_shuffle = True
        if self.use_shuffle:
            in_channels = 12
        else:
            in_channels = 3
        self.out_ch = 32
        self.con_layer = nn.Conv2d(in_channels=in_channels, out_channels=self.out_ch, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1))

        self.stage_layer_1 = Stage_1(self.out_ch)
        self.stage_layer_2 = Stage_2(self.out_ch)

        # aggregate/blend multi-level aligned features
        self.conv_blend_lv1 = nn.Conv2d(args.n_feats * (args.NUM_AUX_FRAMES + 1), args.NUM_AUX_FRAMES + 1, (3, 3),(1, 1), (1, 1), bias=True)
        self.conv_blend_lv2 = nn.Conv2d(args.n_feats * (args.NUM_AUX_FRAMES + 1), args.NUM_AUX_FRAMES + 1, (3, 3),(1, 1), (1, 1), bias=True)
        self.conv_blend_lv3 = nn.Conv2d(args.n_feats * (args.NUM_AUX_FRAMES + 1), args.NUM_AUX_FRAMES + 1, (3, 3),(1, 1), (1, 1), bias=True)

        self.conv_channel_lv1 = nn.Conv2d(args.n_feats, args.n_feats, (1, 1), (1, 1), 0, bias=True)
        self.conv_channel_lv2 = nn.Conv2d(args.n_feats, args.n_feats, (1, 1), (1, 1), (0, 0), bias=True)
        self.conv_channel_lv3 = nn.Conv2d(args.n_feats, args.n_feats, (1, 1), (1, 1), (0, 0), bias=True)


    def load_weight(self , freeze ):
        pre_trained_dir = 'pre_trained/vdm/shuffle_49.pth'
        if not self.use_shuffle:
            pre_trained_dir = 'pre_trained/vdm/noshuffle_49.pth'
            pre_trained_weights = torch.load(pre_trained_dir)['state_dict']
            self.MainNet_V1.load_state_dict(pre_trained_weights, strict=False)
            print('initialize: %s' % pre_trained_dir)
            # freeze the demoireing weights
            if freeze:
                to_freeze_names = pre_trained_weights.keys()
                for name, param in self.MainNet_V1.named_parameters():
                    if name in to_freeze_names:
                        param.requires_grad = False
                print('freeze pre-trained params')

    def down_shuffle(self, x, r):
        b, c, h, w = x.size()
        out_channel = c * (r ** 2)
        out_h = h // r
        out_w = w // r
        x = x.view(b, c, out_h, r, out_w, r)
        out = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)
        out = self.con_layer(out)
        return out

    def save_image(self, image, name):
        image_outpath = "./feats_image_output/"
        try:
            os.makedirs(image_outpath)
        except Exception as e:
            print(e)
        image_outpath = os.path.join(image_outpath , name)
        torchvision.utils.save_image(image.detach().cpu(), image_outpath)

    def blend_type2(self, align_feats_lv1, align_feats_lv2, align_feats_lv3):
        merge_feats_lv1 = align_feats_lv1[:, 0:self.args.n_feats, :, :] * 1 / (1 + self.args.NUM_AUX_FRAMES)
        merge_feats_lv2 = align_feats_lv2[:, 0:self.args.n_feats, :, :] * 1 / (1 + self.args.NUM_AUX_FRAMES)
        merge_feats_lv3 = align_feats_lv3[:, 0:self.args.n_feats, :, :] * 1 / (1 + self.args.NUM_AUX_FRAMES)
        for j in range(1, 1 + self.args.NUM_AUX_FRAMES):
            merge_feats_lv1 = merge_feats_lv1 + align_feats_lv1[:,(self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :,:] * 1 / (1 + self.args.NUM_AUX_FRAMES)
            merge_feats_lv2 = merge_feats_lv2 + align_feats_lv2[:,(self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :,:] * 1 / (1 + self.args.NUM_AUX_FRAMES)
            merge_feats_lv3 = merge_feats_lv3 + align_feats_lv3[:,(self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :,:] * 1 / (1 + self.args.NUM_AUX_FRAMES)
        return merge_feats_lv1, merge_feats_lv2, merge_feats_lv3

    def blend_type1(self, align_feats_lv1, align_feats_lv2, align_feats_lv3):
        weight_lv1 = F.softmax(self.conv_blend_lv1(align_feats_lv1), dim=1)
        weight_lv2 = F.softmax(self.conv_blend_lv2(align_feats_lv2), dim=1)
        weight_lv3 = F.softmax(self.conv_blend_lv3(align_feats_lv3), dim=1)

        # # save blending weights
        # self.save_image(weight_lv3[:,0:1,:,:], 'weight_lv3_0.png')
        # self.save_image(weight_lv2[:,0:1,:,:], 'weight_lv2_0.png')
        # self.save_image(weight_lv1[:,0:1,:,:], 'weight_lv1_0.png')

        # n_feats = 64
        merge_feats_lv1 = align_feats_lv1[:, 0:self.args.n_feats, :, :] * weight_lv1[:, 0:1, :, :]
        merge_feats_lv2 = align_feats_lv2[:, 0:self.args.n_feats, :, :] * weight_lv2[:, 0:1, :, :]
        merge_feats_lv3 = align_feats_lv3[:, 0:self.args.n_feats, :, :] * weight_lv3[:, 0:1, :, :]
        for j in range(1, 1 + self.args.NUM_AUX_FRAMES):
            merge_feats_lv1 = merge_feats_lv1 + align_feats_lv1[:,(self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :,:] * weight_lv1[:, j:(j + 1), :, :]
            merge_feats_lv2 = merge_feats_lv2 + align_feats_lv2[:,(self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :,:] * weight_lv2[:, j:(j + 1), :, :]
            merge_feats_lv3 = merge_feats_lv3 + align_feats_lv3[:,(self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :,:] * weight_lv3[:, j:(j + 1), :, :]

            # save blending weights, auxiliary frames
            # self.save_image(weight_lv3[:,j:(j+1),:,:],'weight_lv3_%s.png' % j)
            # self.save_image(weight_lv2[:,j:(j+1),:,:],'weight_lv2_%s.png' % j)
            # self.save_image(weight_lv1[:,j:(j+1),:,:],'weight_lv1_%s.png' % j)

        return merge_feats_lv1, merge_feats_lv2, merge_feats_lv3

    def forward(self, cur=None, ref=None, label=None, blend=1):
        if self.use_shuffle:
            cur = self.down_shuffle(cur, 2)
        cur_feats = get_multi_level(cur)
        cur_lv1, cur_lv2, cur_lv3 = self.stage_layer(cur)

        # aligned features
        align_feats_lv1 = cur_lv1
        align_feats_lv2 = cur_lv2
        align_feats_lv3 = cur_lv3
        for i in range(self.args.NUM_AUX_FRAMES):
            ref_tmp = ref[:, (0 + 3 * i):(3 + 3 * i), :, :]
            if self.use_shuffle:
                ref_tmp = self.down_shuffle(ref_tmp, 2)
            ref_tmp = get_multi_level(ref_tmp)
            T_lv1, T_lv2, T_lv3 = self.stage_layer_2(ref_tmp, cur_feats)
            align_feats_lv1 = torch.cat((align_feats_lv1, T_lv1), dim=1)
            align_feats_lv2 = torch.cat((align_feats_lv2, T_lv2), dim=1)
            align_feats_lv3 = torch.cat((align_feats_lv3, T_lv3), dim=1)

        if blend == 1:  # use predicted blending weights
            merge_feats_lv1, merge_feats_lv2, merge_feats_lv3 = self.blend_type1(align_feats_lv1, align_feats_lv2,
                                                                                 align_feats_lv3)

        elif blend == 2:
            merge_feats_lv1, merge_feats_lv2, merge_feats_lv3 = self.blend_type2(align_feats_lv1, align_feats_lv2,
                                                                                 align_feats_lv3)

        # refine the merged features
        merge_feats_lv1 = self.conv_channel_lv1(merge_feats_lv1)
        merge_feats_lv2 = self.conv_channel_lv2(merge_feats_lv2)
        merge_feats_lv3 = self.conv_channel_lv3(merge_feats_lv3)

        # demoireing
        dm_lv3, dm_lv2, dm_lv1, f_lv1, f_lv2, f_lv3 = self.MainNet_V1(merge_feats_lv3, merge_feats_lv2, merge_feats_lv1)
        return dm_lv3, dm_lv2, dm_lv1, f_lv1, f_lv2, f_lv3
