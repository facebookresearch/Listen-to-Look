#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
from .networks import weights_init, ClassifierNet, VisualNet, AudioNet
from .audioPreview_model import AudioPreviewModel
from .imageAudio_model import ImageAudioModel
from .imageAudioClassify_model import ImageAudioClassifyModel

import sys
sys.path.insert(0, '..')
from utils.checkpointer import Checkpointer
        
class ModelBuilder():
    # builder for audio preview recurrent network
    def build_audioPreviewLSTM(self, net_imageAudioFeature, net_classifier, args, weights=''):
        net = AudioPreviewModel(net_imageAudioFeature, net_classifier, args)

        if len(weights) > 0:
            print('Loading weights for lstm')
            net.load_state_dict(torch.load(weights))
        return net

    def build_imageAudioClassifierNet(self, net_imageAudio, net_classifier, args, weights=''):
        net = ImageAudioClassifyModel(net_imageAudio, net_classifier, args)
        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading the weights for imageAudioClassifier network')
            checkpointer = Checkpointer(net)
            checkpointer.load_model_only(weights)
        return net

    def build_imageAudioNet(self, weights=''):
        net = ImageAudioModel()
        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading the fc weights for imageAudio network')
            checkpointer = Checkpointer(net)
            checkpointer.load_model_only(weights)
        return net

    # builder for visual stream
    def build_image(self, weights=''):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        net = VisualNet(original_resnet)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net

    def build_audio(self, weights=''):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        net = AudioNet(original_resnet)
        if len(weights) > 0:
            print('Loading weights for audio stream')
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

