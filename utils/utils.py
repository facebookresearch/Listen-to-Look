#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Part of the codes are borrowed or adapted from Zuxuan Wu

import torch
import torchaudio
import random
import numpy as np

def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell

def find_length(list_tensors):
    """find the length of list of tensors"""
    length = [x.shape[0] for x in list_tensors]
    return length

def pad_tensor(tensor, length):
    """Pad a tensor, given by the max length"""
    if tensor.size(0) == length:
        return tensor
    return torch.cat([tensor, torch.zeros([length - tensor.size(0), tensor.size(1)])])
   
def trim_tensor(tensor, start_index, length):
    """trim a tensor, given by the max length"""
    return tensor[start_index:start_index+length, :]

def process_list_tensors(image_features, audio_features, max_length=None, episode_length=None):
    """Pad a list of tensors and return a list of tensors"""
    tensor_length = find_length(image_features)
    #store the indexes that are too short
    tensors_too_short = [index for index in range(len(tensor_length)) if tensor_length[index] < episode_length]

    if max_length is None:
        max_length = max(tensor_length)
    else:
        if max(tensor_length) < max_length:
            max_length = max(tensor_length)
    processed_image_features = []
    processed_audio_features = []
    for i in range(len(image_features)):
        image_tensor = image_features[i]
        audio_tensor = audio_features[i]
        if image_tensor.shape[0] < max_length:
            image_tensor = pad_tensor(image_tensor, max_length)
            audio_tensor = pad_tensor(audio_tensor, max_length)
        elif image_tensor.shape[0] > max_length:
            start_index = random.randint(0, image_tensor.shape[0] - max_length)
            image_tensor = trim_tensor(image_tensor, start_index, max_length)
            audio_tensor = trim_tensor(audio_tensor, start_index, max_length)
        processed_image_features.append(image_tensor)
        processed_audio_features.append(audio_tensor)

    #make sure number of features is larger or equal to episode length, other wise copy the last features
    if len(tensors_too_short) > 0:
        for index in tensors_too_short:
            current_length = tensor_length[index]
            for position in range(episode_length - current_length):
                processed_image_features[index][position + current_length, :] = processed_image_features[index][current_length-1, :]          
                processed_audio_features[index][position + current_length, :] = processed_audio_features[index][current_length-1, :]          

    return torch.stack(processed_image_features), torch.stack(processed_audio_features), tensor_length, max_length

def create_mask(batchsize, max_length, episode_length, length):
    """Given the length create a mask given a padded tensor"""
    tensor_mask = torch.zeros(batchsize, max_length)
    for idx, row in enumerate(tensor_mask):
        row[:max(episode_length,length[idx])] = 1
    return tensor_mask