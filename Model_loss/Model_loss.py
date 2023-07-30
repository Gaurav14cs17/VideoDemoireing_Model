import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


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

        self.blocks = torch.nn.ModuleList(self.blocks)

        self.transform = torch.nn.functional.interpolate

        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def get_resize(self, inputs, target):
        inputs = self.transform(inputs, mode='bilinear', size=(224, 224), align_corners=False)
        target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        return inputs, target

    def get_norm_image(self , input , target ):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        return input , target

    def get_3_ch_image(self , image ):
        image = image.repeat(1, 3, 1, 1)
        return image

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[], mask=None, return_feature=False):
        if input.shape[1] != 3:
            input = self.get_3_ch_image(input)
            target = self.get_3_ch_image(target)

        input , target = self.get_norm_image(input , target)
        if self.resize:
            input , target = self.get_resize(input , target)

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
