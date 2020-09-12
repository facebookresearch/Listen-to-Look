#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)
    
def build_mlp(input_dim, hidden_dims, output_dim=None, use_batchnorm=False, use_relu=True, dropout=0):
    layers = []
    D = input_dim
    if hidden_dims:
        for dim in hidden_dims:
            layers.append(nn.Linear(D, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            D = dim
    if output_dim:
        layers.append(nn.Linear(D, output_dim))
    return nn.Sequential(*layers)

class VisualNet(nn.Module):
    def __init__(self, original_resnet):
        super(VisualNet, self).__init__()
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers) #features before conv1x1

    def forward(self, x):
        x = self.feature_extraction(x)
        x = F.adaptive_avg_pool2d(x, (1,1)) #RG: pool the features spatially
        x = x.view(x.size(0), -1)
        return x

class AudioNet(nn.Module):
    def __init__(self, original_resnet):
        super(AudioNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.apply(weights_init)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers) #features before conv1x1
        self.freq_conv1x1 = nn.Conv2d(3, 1, kernel_size=(1,1), padding=0, stride=1) #freq dimen = 5
        self.freq_conv1x1.apply(weights_init)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.permute(0,3,1,2) #switch the channel dimension
        x = self.freq_conv1x1(x) #collapse freq dimension
        x = x.permute(0,2,1,3) #switch it back        
        x = F.adaptive_max_pool2d(x, (1,1)) #RG: pool the features temporally
        x = x.view(x.size(0), -1)
        return x

class ClassifierNet(nn.Module):
    def __init__(self, input_dim, finetune_classes):
        super(ClassifierNet, self).__init__()
        self.classifier = torch.nn.Linear(input_dim, finetune_classes)

    def forward(self, x):
        x = self.classifier(x)
        return x
