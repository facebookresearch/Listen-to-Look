#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
from .networks import weights_init, ClassifierNet
from .audioPreview_model import AudioPreviewModel
from .imageAudioClassify_model import ImageAudioClassifyModel

import sys
sys.path.insert(0, '..')
from utils.checkpointer import Checkpointer
        
class ModelBuilder():
    # builder for audio preview recurrent network
    def build_audioPreviewLSTM(self, net_classifier, args, weights=''):
        net = AudioPreviewModel(net_classifier, args)

        if len(weights) > 0:
            print('Loading weights for lstm')
            net.load_state_dict(torch.load(weights))
        return net


    def build_imageAudioClassifierNet(self, net_classifier, args, weights=''):
        net = ImageAudioClassifyModel(net_classifier, args)
        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading the weights for imageAudioClassifier network')
            checkpointer = Checkpointer(net)
            checkpointer.load_model_only(weights)
        return net


    def build_classifierNet(self, input_dims=512, num_classes=200, weights='', ):
        net = ClassifierNet(input_dims, num_classes)
        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for classifier')
            checkpointer = Checkpointer(net)
            checkpointer.load_model_only(weights)
        return net

