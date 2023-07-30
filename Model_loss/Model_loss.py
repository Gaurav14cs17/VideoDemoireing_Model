import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

'''
DISTS_Loss : https://github.com/dingkeyan93/DISTS/blob/master/DISTS_pytorch/DISTS_pt.py
'''


class VGG_Perceptual_Loss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGG_Perceptual_Loss, self).__init__()
        self.model = torchvision.models.vgg16_bn(pretrained=True)
        self.model.eval()

        self.blocks = []
        self.blocks.append(self.model.features[:4])
        self.blocks.append(self.model.features[4:9])
        self.blocks.append(self.model.features[9:16])
        self.blocks.append(self.model.features[16:23])

        for blk in self.blocks:
            for fea in blk:
                fea.requires_grad = False

        self.blocks = nn.ModuleList(self.blocks)

        self.transform = nn.functional.interpolate

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        '''
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        '''

        '''
        self.chns = [3,64,128,256,512,512]
        
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        '''

        self.resize = resize

    def get_resize(self, inputs, target):
        inputs = self.transform(inputs, mode='bilinear', size=(224, 224), align_corners=False)
        target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        return inputs, target

    def get_norm_image(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        return input, target

    def get_3_ch_image(self, image):
        image = image.repeat(1, 3, 1, 1)
        return image

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[], mask=None, return_feature=False):
        if input.shape[1] != 3:
            input = self.get_3_ch_image(input)
            target = self.get_3_ch_image(target)

        input, target = self.get_norm_image(input, target)
        if self.resize:
            input, target = self.get_resize(input, target)

        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                if mask is not None:
                    _, _, H, W = x.shape
                    mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')[:, 0:1, :, :]
                    x = x * mask_resized
                    y = y * mask_resized
                    loss += torch.nn.functional.l1_loss(x, y)
                else:
                    loss += torch.nn.functional.l1_loss(x, y)

                if return_feature:
                    return x, y

            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


'''
Single_Perceptual_loss
'''


class single_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam_p=1.0, lam_l=0.5):
        super(single_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGG_Perceptual_Loss()
        self.lam_p = lam_p
        self.lam_l = lam_l

    def forward(self, out, gt, feature_layers=[2], mask=None):
        if mask is not None:
            loss = self.lam_p * self.loss_fn(out, gt, feature_layers=feature_layers,
                                             mask=mask) + self.lam_l * F.l1_loss(out * mask, gt * mask)
        else:
            loss = self.lam_p * self.loss_fn(out, gt, feature_layers=feature_layers) + self.lam_l * F.l1_loss(out, gt)
        return loss


'''
Multi Perceptual_loss
'''


class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam_p=1.0, lam_l=0.5):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGG_Perceptual_Loss()
        self.lam_p = lam_p
        self.lam_l = lam_l

    def get_resize_image(self, image, scale_factor):
        return F.interpolate(image, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, out3, out2, out1, gt1, feature_layers=[2], mask=None):
        gt2 = self.get_resize_image(gt1, scale_factor=0.5)
        gt3 = self.get_resize_image(gt1, scale_factor=0.25)
        if mask is not None:
            mask2 = self.get_resize_image(mask, scale_factor=0.5)
            mask3 = self.get_resize_image(mask, scale_factor=0.25)
            loss1 = self.lam_p * self.loss_fn(out1, gt1, feature_layers=feature_layers,
                                              mask=mask) + self.lam_l * F.l1_loss(out1 * mask, gt1 * mask)
            loss2 = self.lam_p * self.loss_fn(out2, gt2, feature_layers=feature_layers,
                                              mask=mask2) + self.lam_l * F.l1_loss(out2 * mask2, gt2 * mask2)
            loss3 = self.lam_p * self.loss_fn(out3, gt3, feature_layers=feature_layers,
                                              mask=mask3) + self.lam_l * F.l1_loss(out3 * mask3, gt3 * mask3)
        else:
            loss1 = self.lam_p * self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam_l * F.l1_loss(out1,
                                                                                                                 gt1)
            loss2 = self.lam_p * self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam_l * F.l1_loss(out2,
                                                                                                                 gt2)
            loss3 = self.lam_p * self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam_l * F.l1_loss(out3,
                                                                                                                 gt3)
        return loss1 + loss2 + loss3
