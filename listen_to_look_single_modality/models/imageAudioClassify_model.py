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

class ImageAudioClassifyModel(torch.nn.Module):
    def name(self):
        return 'ImageAudioClassifyModel'

    def __init__(self, net_classifier, args):
        super(ImageAudioClassifyModel, self).__init__()
        self.args = args
        #initialize model
        self.net_classifier = net_classifier

    def forward(self, imageAudio_embedding):
        predictions = self.net_classifier(imageAudio_embedding)
        if self.args.feature_extraction:
            return imageAudio_embedding, predictions
        return predictions
