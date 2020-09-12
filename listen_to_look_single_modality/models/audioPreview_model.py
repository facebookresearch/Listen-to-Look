#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Part of the codes are adapted from Zuxuan Wu

import os
import numpy as np
import torch
import random
from torch import optim
import torch.nn.functional as F
from . import networks
from torch.autograd import Variable
from utils.checkpointer import Checkpointer
from utils.utils import init_hidden

class AudioPreviewModel(torch.nn.Module):
    def name(self):
        return 'AudioPreviewModel'

    def __init__(self, net_classifier, args):
        super(AudioPreviewModel, self).__init__()
        self.args = args
        self.num_classes = args.num_classes
        self.with_replacement = args.with_replacement
        self.imageKey = args.imageKey #using image features as key
        self.net_classifier = net_classifier
        self.hidden_size = 1024
        self.rnn_input_size = self.args.embedding_size
        self.key_query_dim = 512
        self.num_layers = 1
        self.bidirectional = False

        queryfeature_kwargs = {
            'input_dim': self.hidden_size,
            'hidden_dims': (self.hidden_size,),
            'output_dim' : self.key_query_dim,
            'use_batchnorm': True,
            'use_relu': True,
            'dropout': 0,
        }

        self.queryfeature_mlp = networks.build_mlp(**queryfeature_kwargs)
        self.queryfeature_mlp.apply(networks.weights_init)

        self.prediction_fc = torch.nn.Linear(self.hidden_size, self.num_classes)
        self.prediction_fc.apply(networks.weights_init)

        self.key_conv1x1 = networks.create_conv(self.rnn_input_size, self.key_query_dim, 1, 0)
        self.key_conv1x1.apply(networks.weights_init)
        
        self.rnn = torch.nn.LSTMCell(input_size=self.rnn_input_size,
                               hidden_size=self.hidden_size, bias=True)

        #adding weight normalization 
        self.rnn = torch.nn.utils.weight_norm(self.rnn, 'weight_hh')
        self.rnn = torch.nn.utils.weight_norm(self.rnn, 'weight_ih')

        self.rnn.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
        self.rnn.bias_hh.data.fill_(0) # initializing the lstm bias with zeros

        self.softmax = torch.nn.Softmax(dim=-1)

    def forwardOne(self, features, feature_banks, feature_masks, hx, dropmask=None):
        h_next, c_next = self.rnn(features, hx)
        query_feature = self.queryfeature_mlp(h_next)

        #prediction at this step
        step_prediction = self.prediction_fc(h_next)

        #get query scores by dot product attention
        query_scores = torch.bmm(query_feature.unsqueeze(1), self.feature_keys).squeeze() / np.sqrt(self.key_query_dim)
        query_scores = query_scores * feature_masks + (1 - feature_masks) * -(1e35) #assign a very small value for padded positions
        normalized_query_scores = self.softmax(query_scores)

        #select indexes for next round based on query scores
        feature_indexes_next = torch.argmax(normalized_query_scores, dim=1)

        #weighted sum of feature banks to generate c_next
        features_next = torch.bmm(feature_banks.permute(0,2,1), normalized_query_scores.unsqueeze(-1)).squeeze(-1)

        return h_next, c_next, features_next, feature_indexes_next, step_prediction

    def forward(self, features, input_gt_predictions, feature_masks, episode_length, use_gt_feature=False, validation=False):
        self.batch_size = features.size(0)
        h_t = init_hidden(self.batch_size, self.hidden_size)
        c_t = init_hidden(self.batch_size, self.hidden_size)
        hx = (h_t, c_t)
        helper_tensor = torch.ones(feature_masks.shape).cuda()

        if self.args.feature_interpolate:
            subsampled_features = features[:, ::self.args.subsample_factor, :]
            feature_banks = F.interpolate(subsampled_features.permute(0,2,1), size=features.shape[1], mode='linear').permute(0,2,1)
            feature_banks = feature_banks * feature_masks.unsqueeze(-1).expand_as(feature_banks)
        else:
            feature_banks = features

        self.feature_keys = self.key_conv1x1(features.permute(0,2,1).unsqueeze(-1)).squeeze(-1)

        #initialze the starting features and indexes
        starting_random_tensor = torch.rand(feature_masks.shape).cuda() + feature_masks #a helper tensor to select the starting position for each video
        feature_indexes_next = torch.argmax(starting_random_tensor, dim=1)
        #feature_indexes_next = torch.zeros([self.batch_size]).long().cuda() #using the first features as start for all
        if self.args.mean_feature_as_start:
            gt_features_next = torch.sum(features, dim=1) / torch.sum(feature_masks, dim=1).unsqueeze(-1)
        else:
            gt_features_next = features[torch.LongTensor(range(self.batch_size)), feature_indexes_next]

        selected_feature_indexes = [] #intialize list to store all feature indexes
        selected_gt_features = [] #initialize list to store gt features
        selected_gt_predictions = [] #intialize list to store all gt predictions
        selected_step_predictions = [] #intialize list to store all step predictions

        for step in range(episode_length):
            #ensure that the frame is not selected in the later iterations
            if self.with_replacement:
                helper_tensor[torch.LongTensor(range(self.batch_size)), feature_indexes_next] = 0 #change to 0 of indexed positions
                feature_masks = feature_masks * helper_tensor
                helper_tensor[torch.LongTensor(range(self.batch_size)), feature_indexes_next] = 1 #change back to 1 to recover all ones

            h_next, c_next, gt_features_next, feature_indexes_next, step_prediction = self.forwardOne(gt_features_next, feature_banks, feature_masks, hx)

            selected_feature_indexes.append(feature_indexes_next)
            selected_gt_prediction = input_gt_predictions[torch.LongTensor(range(self.batch_size)), feature_indexes_next]
            selected_gt_predictions.append(selected_gt_prediction)

            if step >= 0: #skip the first one
                selected_gt_features.append(gt_features_next)
                selected_step_predictions.append(step_prediction)

            hx_next = (h_next, c_next)
            hx = hx_next

            if use_gt_feature:
                gt_features_next = feature_banks[torch.LongTensor(range(self.batch_size)), feature_indexes_next]
                
            if torch.sum(feature_masks) == 0:
                break

        selected_feature_indexes = torch.stack(selected_feature_indexes).permute(1,0) #permute to recover batch dimension
        selected_gt_predictions = torch.stack(selected_gt_predictions).permute(1,2,0)  #permute to recover batch dimension
        selected_gt_features = torch.stack(selected_gt_features).permute(1,2,0)
        predicted_features = torch.mean(selected_gt_features, dim=2)
        gt_predictions = torch.mean(selected_gt_predictions, dim=2)
        predictions = self.net_classifier(predicted_features)
        return predictions, gt_predictions, selected_step_predictions
