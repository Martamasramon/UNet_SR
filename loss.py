#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:56:57 2021

@author: quentin
"""

import torchvision.models as models
import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_msssim     import ssim
from torchvision.transforms import Normalize

def ssim_loss(pred, target):
    return 1 - ssim(pred, target, data_range=1, size_average=True) 

def transform_perceptual(img):
    transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.repeat(1, 3, 1, 1)
    img = transform(img)
    return img

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(weights='DEFAULT').features[:4].to(device).eval())
        blocks.append(models.vgg16(weights='DEFAULT').features[4:9].to(device).eval())
        blocks.append(models.vgg16(weights='DEFAULT').features[9:16].to(device).eval())
        blocks.append(models.vgg16(weights='DEFAULT').features[16:23].to(device).eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.device=device
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to(device)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to(device)
        self.resize = resize

    def forward(self, input, target):
        input  = transform_perceptual(input)
        target = transform_perceptual(target)
        
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1).to(self.device)
            target = target.repeat(1, 3, 1, 1).to(self.device)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + nn.MSELoss()(x, y)
        return loss
    
def PSNR(Img_pred, Img_true):
    return 10 * torch.log10(torch.max(Img_pred)**2 / nn.MSELoss()(Img_pred,Img_true))

def cosine_contrastive_loss(x, y):
    # Cosine contrastive loss. Assumes x and y are normalized or raw embeddings
    return 1 - F.cosine_similarity(x, y, dim=1).mean()