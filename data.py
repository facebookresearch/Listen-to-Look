#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Part of the codes are borrowed or adapted from Zuxuan Wu and Bruno Korbar

import torch
from torchvision.datasets import DatasetFolder
import time
import pickle
import numpy as np
import random
from utils.utils import process_list_tensors, create_mask

class AudioVisualDataset(DatasetFolder):
    def __init__(self, args, dataset_file=None, logger=None):
        self.args = args
        t = time.time()
        self.feature_paths, self.labels = pickle.load(open(dataset_file, "rb"))

        if logger is not None:
            logger.debug('Time to load paths: {}'.format(time.time() - t))
            logger.debug("Number of videos: {}".format(len(self.feature_paths)))
       
    def __getitem__(self, idx):
        #path to load features
        feature_path = self.feature_paths[idx] # path to the imageAudio feature
        label = self.labels[idx]
        image_feature_path = feature_path.replace('imageAudio_features', 'image_features') # path to the image feature
        audio_feature_path = feature_path.replace('imageAudio_features', 'audio_features') # path to the audio feature
        
        #load features
        image_feature = np.load(image_feature_path)
        audio_feature = np.load(audio_feature_path)

        if self.args.feature_subsample:
            image_feature = image_feature[::self.args.subsample_factor, :]
            audio_feature = audio_feature[::self.args.subsample_factor, :]

        image_feature = torch.Tensor(image_feature)
        audio_feature = torch.Tensor(audio_feature)

        return image_feature, audio_feature, label, idx
        
    def __len__(self):
        return len(self.feature_paths)

def create_training_dataset(args, logger=None):
    collate = feature_collate_train
    train_ds = AudioVisualDataset(args, dataset_file=args.train_dataset_file, logger=logger)
    return train_ds, collate

def create_validation_dataset(args, logger=None):
    collate = feature_collate_val
    val_ds = AudioVisualDataset(args, dataset_file=args.test_dataset_file, logger=logger)
    return val_ds, collate

#Collate functions 
def feature_collate_train(batch):
    """
    A collate function for data loading
    Return: padded tensors
    """
    #unpack into a list
    max_length = random.randint(200,500) #randomly set a max length of clip length for training
    episode_length = 10
    data = list(zip(*batch))
    processed_imgfeat, processed_audiofeat, vid_length, max_length = process_list_tensors(data[0], data[1], max_length, episode_length=episode_length)
    feat_mask = create_mask(len(vid_length), max_length, episode_length, vid_length)
    labels = torch.LongTensor(np.array(data[2]))
    idxs = np.array(data[3])
    return processed_imgfeat, processed_audiofeat, feat_mask, labels, idxs

def feature_collate_val(batch):
    """
    A collate function for data loading
    Return: padded tensors
    """
    #unpack into a list
    episode_length = 10
    data = list(zip(*batch))
    processed_imgfeat, processed_audiofeat, vid_length, max_length = process_list_tensors(data[0], data[1], episode_length=episode_length)
    feat_mask = create_mask(len(vid_length), max_length, episode_length, vid_length)
    labels = torch.LongTensor(np.array(data[2]))
    idxs = np.array(data[3])
    return processed_imgfeat, processed_audiofeat, feat_mask, labels, idxs
