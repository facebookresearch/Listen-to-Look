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
from torch import optim
import torch.nn.functional as F
from . import networks
from torch.autograd import Variable
from utils.checkpointer import Checkpointer
from utils.utils import init_hidden

class AudioPreviewModel(torch.nn.Module):
    def name(self):
        return 'AudioPreviewModel'

    def __init__(self, net_imageAudio, net_classifier, args):
        super(AudioPreviewModel, self).__init__()
        #initialize model
        self.args = args
        self.num_classes = args.num_classes
        self.with_replacement = args.with_replacement

        self.net_imageAudio = net_imageAudio
        self.net_classifier = net_classifier
        self.hidden_size = 1024
        self.rnn_input_size = 512
        self.key_query_dim = 512
        self.num_layers = 1
        self.bidirectional = False

        queryfeature_kwargs = {
            'input_dim': self.hidden_size,
            'hidden_dims': (1024,),
            'output_dim' : self.key_query_dim,
            'use_batchnorm': True,
            'use_relu': True,
            'dropout': 0,
        }

        self.image_queryfeature_mlp = networks.build_mlp(**queryfeature_kwargs)
        self.image_queryfeature_mlp.apply(networks.weights_init)

        self.audio_queryfeature_mlp = networks.build_mlp(**queryfeature_kwargs)
        self.audio_queryfeature_mlp.apply(networks.weights_init)

        self.prediction_fc = torch.nn.Linear(self.hidden_size, self.num_classes)
        self.prediction_fc.apply(networks.weights_init)

        self.image_key_conv1x1 = networks.create_conv(self.rnn_input_size, self.key_query_dim, 1, 0)
        self.image_key_conv1x1.apply(networks.weights_init)

        self.audio_key_conv1x1 = networks.create_conv(self.rnn_input_size, self.key_query_dim, 1, 0)
        self.audio_key_conv1x1.apply(networks.weights_init)

        self.rnn = torch.nn.LSTMCell(input_size=self.rnn_input_size,
                               hidden_size=self.hidden_size, bias=True)

        #adding weight normalization 
        self.rnn = torch.nn.utils.weight_norm(self.rnn, 'weight_hh')
        self.rnn = torch.nn.utils.weight_norm(self.rnn, 'weight_ih')

        self.rnn.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
        self.rnn.bias_hh.data.fill_(0) # initializing the lstm bias with zeros

        self.modality_attention = torch.nn.Linear(self.hidden_size, 2)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forwardOne(self, image_features, audio_features, image_feature_banks, audio_feature_banks, image_feature_masks, audio_feature_masks, hx):
        imageAudio_features = self.net_imageAudio(image_features, audio_features)
        h_next, c_next = self.rnn(imageAudio_features, hx)
        image_query_feature = self.image_queryfeature_mlp(h_next)
        audio_query_feature = self.audio_queryfeature_mlp(h_next)

        #prediction at this step
        step_prediction = self.prediction_fc(h_next)

        #get query scores by dot product attention
        image_query_scores = torch.bmm(image_query_feature.unsqueeze(1), self.image_feature_keys).squeeze(1) / np.sqrt(self.key_query_dim)
        audio_query_scores = torch.bmm(audio_query_feature.unsqueeze(1), self.audio_feature_keys).squeeze(1) / np.sqrt(self.key_query_dim)        
        image_query_scores = image_query_scores * image_feature_masks + (1 - image_feature_masks) * -(1e35) #assign a very small value for padded positions
        audio_query_scores = audio_query_scores * audio_feature_masks + (1 - audio_feature_masks) * -(1e35) #assign a very small value for padded positions

        #apply softmax to normalize query scores
        normalized_image_query_scores = self.softmax(image_query_scores)
        normalized_audio_query_scores = self.softmax(audio_query_scores)

        #select indexes based on query scores
        image_feature_indexes_next = torch.argmax(normalized_image_query_scores, dim=1)
        audio_feature_indexes_next = torch.argmax(normalized_audio_query_scores, dim=1)

        #weighted sum of the feature banks to generate the next input feature
        image_features_next_from_image_query_scores = torch.bmm(image_feature_banks.permute(0,2,1), normalized_image_query_scores.unsqueeze(-1)).squeeze(-1)
        image_features_next_from_audio_query_scores = torch.bmm(image_feature_banks.permute(0,2,1), normalized_audio_query_scores.unsqueeze(-1)).squeeze(-1)
        audio_features_next_from_image_query_scores = torch.bmm(audio_feature_banks.permute(0,2,1), normalized_image_query_scores.unsqueeze(-1)).squeeze(-1)
        audio_features_next_from_audio_query_scores = torch.bmm(audio_feature_banks.permute(0,2,1), normalized_audio_query_scores.unsqueeze(-1)).squeeze(-1)

        #predict the modality attention weight
        modality_attention_scores = self.softmax(self.modality_attention(h_next))

        #aggregate based one modality attention weight
        image_features_next = image_features_next_from_image_query_scores * modality_attention_scores[:,0].unsqueeze(-1).expand_as(image_features) \
                            + image_features_next_from_audio_query_scores * modality_attention_scores[:,1].unsqueeze(-1).expand_as(image_features)
        audio_features_next = audio_features_next_from_image_query_scores * modality_attention_scores[:,0].unsqueeze(-1).expand_as(image_features) \
                            + audio_features_next_from_audio_query_scores * modality_attention_scores[:,1].unsqueeze(-1).expand_as(image_features)

        return h_next, c_next, imageAudio_features, image_features_next, audio_features_next, image_feature_indexes_next, audio_feature_indexes_next, step_prediction, modality_attention_scores

    def forward(self, image_features, audio_features, feature_masks, episode_length, validate=False):
        self.batch_size = image_features.size(0)
        h_t = init_hidden(self.batch_size, self.hidden_size)
        c_t = init_hidden(self.batch_size, self.hidden_size)
        hx = (h_t, c_t)
        helper_tensor = torch.ones(feature_masks.shape).cuda()
        audio_feature_masks = feature_masks.clone()

        if self.args.feature_interpolate:
            subsampled_image_features = image_features[:, ::self.args.subsample_factor, :]
            subsampled_audio_features = audio_features[:, ::self.args.subsample_factor, :]
            image_feature_banks = F.interpolate(subsampled_image_features.permute(0,2,1), size=image_features.shape[1], mode='linear').permute(0,2,1)
            audio_feature_banks = F.interpolate(subsampled_audio_features.permute(0,2,1), size=audio_features.shape[1], mode='linear').permute(0,2,1)
            image_feature_banks = image_feature_banks * feature_masks.unsqueeze(-1).expand_as(image_feature_banks)
            audio_feature_banks = audio_feature_banks * audio_feature_masks.unsqueeze(-1).expand_as(audio_feature_banks)
        else:
            image_feature_banks = image_features
            audio_feature_banks = audio_features

        self.image_feature_keys = self.image_key_conv1x1(image_feature_banks.permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        self.audio_feature_keys = self.audio_key_conv1x1(audio_feature_banks.permute(0,2,1).unsqueeze(-1)).squeeze(-1)

        #initialze the starting features and indexes
        starting_random_tensor = torch.rand(feature_masks.shape).cuda() + feature_masks #a helper tensor to select the starting position for each video
        image_feature_indexes_next = torch.argmax(starting_random_tensor, dim=1)
        audio_feature_indexes_next = image_feature_indexes_next

        if self.args.mean_feature_as_start:
            image_features_next = torch.sum(image_feature_banks, dim=1) / torch.sum(feature_masks, dim=1).unsqueeze(-1)
            audio_features_next = torch.sum(audio_feature_banks, dim=1) / torch.sum(audio_feature_masks, dim=1).unsqueeze(-1)
        else:
            image_features_next = image_features[torch.LongTensor(range(self.batch_size)), image_feature_indexes_next]
            audio_features_next = audio_features[torch.LongTensor(range(self.batch_size)), audio_feature_indexes_next]

        selected_image_feature_indexes = [] #intialize list to store all feature indexes selected by querying image features
        selected_audio_feature_indexes = [] #intialize list to store all feature indexes selected by querying audio features
        selected_imageAudioFeatures = [] #intialize list to store all imageAudio features
        selected_step_predictions = [] #intialize list to store predictions at all steps

        for step in range(episode_length):
            #ensure that the index is not selected in the later iterations
            if self.with_replacement:
                helper_tensor[torch.LongTensor(range(self.batch_size)), image_feature_indexes_next] = 0 #change to 0 of indexed positions
                feature_masks = feature_masks * helper_tensor
                helper_tensor[torch.LongTensor(range(self.batch_size)), image_feature_indexes_next] = 1 #change back to 1 to recover all ones
                helper_tensor[torch.LongTensor(range(self.batch_size)), audio_feature_indexes_next] = 0 #change to 0 of indexed positions
                audio_feature_masks = audio_feature_masks * helper_tensor
                helper_tensor[torch.LongTensor(range(self.batch_size)), audio_feature_indexes_next] = 1 #change back to 1 to recover all ones

            h_next, c_next, imageAudioFeatures, image_features_next, audio_features_next, image_feature_indexes_next, audio_feature_indexes_next, step_prediction, _ = self.forwardOne(image_features_next, audio_features_next, image_feature_banks, audio_feature_banks, feature_masks, audio_feature_masks, hx)

            selected_image_feature_indexes.append(image_feature_indexes_next)
            selected_audio_feature_indexes.append(audio_feature_indexes_next)
            selected_imageAudioFeatures.append(imageAudioFeatures)
            selected_step_predictions.append(step_prediction)

            hx_next = (h_next, c_next)
            hx = hx_next

            if validate:
                image_features_next = image_feature_banks[torch.LongTensor(range(self.batch_size)), image_feature_indexes_next]
                audio_features_next = audio_feature_banks[torch.LongTensor(range(self.batch_size)), image_feature_indexes_next]

        selected_imageAudioFeatures = torch.stack(selected_imageAudioFeatures).permute(1,2,0) #permute to recover batch dimension, and in order of batch x feature_dim x num_of_features  
        selected_image_feature_indexes = torch.stack(selected_image_feature_indexes).permute(1,0) #permute to recover batch dimension
        selected_audio_feature_indexes = torch.stack(selected_audio_feature_indexes).permute(1,0) #permute to recover batch dimension
        predicted_features = torch.mean(selected_imageAudioFeatures, dim=2)
        predictions = self.net_classifier(predicted_features)
                
        return predictions, selected_imageAudioFeatures, selected_step_predictions