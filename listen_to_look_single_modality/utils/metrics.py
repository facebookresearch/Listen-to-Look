#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Part of the codes are borrowed or adapted from Bruno Korbar

import torch
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.special import softmax
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class NaiveMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        self.sum = 0
        self.values = {}

    def update(self, val, idxs):
        self.val = val.float().mean()
        n = len(idxs)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        for v, idx in zip(val.tolist(), idxs.tolist()):
            self.values[idx] = v

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def calculate_accuracy(outputs, targets, accumulate=True):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)

    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    if not accumulate:
        return correct.squeeze(0)
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size

def aggredate_video_accuracy(softmaxes, labels, topk=(1,), aggregate="mean"):
    maxk = max(topk)
    output_batch = torch.stack(
        [torch.mean(torch.stack(
            softmaxes[sms]),
            0,
            keepdim=False
        ) for sms in softmaxes.keys()])
    num_videos = output_batch.size(0)
    output_labels = torch.stack(
        [labels[video_id] for video_id in softmaxes.keys()])

    _, pred = output_batch.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(output_labels.expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / num_videos))
    return res

def aggredate_video_map(softmaxes, labels):
    output_batch = torch.stack(
        [torch.mean(torch.stack(
            softmaxes[sms]),
            0,
            keepdim=False
        ) for sms in softmaxes.keys()])
    num_videos = output_batch.size(0)
    output_labels = torch.stack(
        [labels[video_id] for video_id in softmaxes.keys()]).data.cpu().numpy()
    num_classes = output_batch[0].size(0)
    output_batch = output_batch.data.cpu().numpy()

    gt_labels = np.zeros([num_videos, num_classes])
    predicted_scores = np.zeros([num_videos, num_classes])
    for i in range(num_videos):
        gt_labels[i,output_labels[i]] = 1
        predicted_scores[i,:] = softmax(output_batch[i])
    
    mean_ap = average_precision_score(gt_labels, predicted_scores)
    return mean_ap

def get_rank():
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()