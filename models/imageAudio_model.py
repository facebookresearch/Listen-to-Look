#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import networks
from torch.autograd import Variable

class ImageAudioModel(torch.nn.Module):
    def name(self):
        return 'ImageAudioModel'

    def __init__(self):
        super(ImageAudioModel, self).__init__()
        #initialize model
        self.imageAudio_fc1 = torch.nn.Linear(512 * 2, 512 * 2)
        self.imageAudio_fc1.apply(networks.weights_init)
        self.imageAudio_fc2 = torch.nn.Linear(512 * 2, 512)
        self.imageAudio_fc2.apply(networks.weights_init)


    def forward(self, image_features, audio_features):
        audioVisual_features = torch.cat((image_features, audio_features), dim=1)
        imageAudio_embedding = self.imageAudio_fc1(audioVisual_features)
        imageAudio_embedding = F.relu(imageAudio_embedding)
        imageAudio_embedding = self.imageAudio_fc2(imageAudio_embedding)
        return imageAudio_embedding