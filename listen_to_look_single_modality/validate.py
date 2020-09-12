#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Part of the codes are taken or adapted from Bruno Korbar

import time
import glob
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from models.models import ModelBuilder
from models.imageAudioClassify_model import ImageAudioClassifyModel

from opts import get_parameters
from utils.checkpointer import Checkpointer

from utils.metrics import AverageMeter, NaiveMeter, calculate_accuracy, aggredate_video_accuracy, aggredate_video_map
from data import create_validation_dataset

from utils.logging import setup_logger

def validate(args, epoch, val_loader, model, criterion, epoch_logger=None,
             writer=None, val_ds=None):
    batch_time = AverageMeter()
    inference_time = AverageMeter()
    avgpool_ce_losses = AverageMeter()
    lstm_ce_losses = AverageMeter()
    losses = AverageMeter()
    avgpool_accuracies = NaiveMeter()
    lstm_accuracies = NaiveMeter()
    gtprediction_accuracies = NaiveMeter()

    if args.compute_mAP:
        avgpool_softmaxes_dic = {}
        lstm_softmaxes_dic = {}
        gtprediction_softmaxes_dic = {}
        labels_dic = {}

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            gt_features, gt_predictions, feature_masks, labels, idx = data
            gt_features = gt_features.cuda()
            feature_masks = feature_masks.cuda()
            labels = labels.cuda()
            batch_size = gt_features.shape[0]
   
            inference_start = time.time()
            predictions, selected_gt_predictions, selected_step_predictions = model.forward(gt_features, gt_predictions, feature_masks, args.episode_length, use_gt_feature=args.gt_feature_eval, validation=True)
            inference_time.update(time.time() - inference_start)
            
            #get mAP
            if args.compute_mAP:
                for j in range(len(idx)):
                    # id of the video file
                    video_id = idx[j]
                    # associated softmax
                    lstm_sm = selected_step_predictions[-1][j]
                    gtprediction_sm = selected_gt_predictions[j]
                    avgpool_sm = predictions[j]
                    label = labels[j]
                    # append it to video dict
                    lstm_softmaxes_dic.setdefault(video_id, []).append(lstm_sm)
                    gtprediction_softmaxes_dic.setdefault(video_id, []).append(gtprediction_sm)
                    avgpool_softmaxes_dic.setdefault(video_id, []).append(avgpool_sm)
                    labels_dic[video_id] = label

            lstm_ce_loss = criterion['CrossEntropyLoss'](selected_step_predictions[-1], labels)  
            avgpool_ce_loss = criterion['CrossEntropyLoss'](predictions, labels)

            #final loss to use
            loss = 0
            if args.with_avgpool_ce_loss:
                loss = loss + avgpool_ce_loss
            if args.with_lstm_ce_loss:
                loss = loss + lstm_ce_loss

            lstm_acc = calculate_accuracy(selected_step_predictions[-1], labels, accumulate=False)
            gtprediction_acc = calculate_accuracy(selected_gt_predictions, labels, accumulate=False)
            avgpool_acc = calculate_accuracy(predictions, labels, accumulate=False)

            avgpool_ce_losses.update(avgpool_ce_loss.data.item(), batch_size)
            lstm_ce_losses.update(lstm_ce_loss.data.item(), batch_size)
            losses.update(loss.data.item(), batch_size)
            avgpool_accuracies.update(avgpool_acc, idx)
            lstm_accuracies.update(lstm_acc, idx)
            gtprediction_accuracies.update(gtprediction_acc, idx)
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Inference Time {inference_time.val:.3f} ({inference_time.avg:.3f})\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Avgpool CE Loss {avgpool_ce_loss.val:.4f} ({avgpool_ce_loss.avg:.4f})\t'
                      'LSTM CE Loss {lstm_ce_loss.val:.4f} ({lstm_ce_loss.avg:.4f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Avgpool Acc {avgpool_acc.val:.3f} ({avgpool_acc.avg:.3f})\t'
                      'LSTM Acc {lstm_acc.val:.3f} ({lstm_acc.avg:.3f})\t'
                      'GT Prediction Acc {gtprediction_acc.val:.3f} ({gtprediction_acc.avg:.3f})\n'.format(
                        i, len(val_loader),
                        inference_time=inference_time,
                        batch_time=batch_time,
                        avgpool_ce_loss=avgpool_ce_losses,
                        lstm_ce_loss=lstm_ce_losses,
                        loss=losses,
                        avgpool_acc=avgpool_accuracies,
                        lstm_acc=lstm_accuracies,
                        gtprediction_acc=gtprediction_accuracies))

    avgpool_acc_per_image = avgpool_accuracies.values
    gtprediction_acc_per_image = gtprediction_accuracies.values
    lstm_acc_per_image = lstm_accuracies.values

    avgpool_final_acc = float(sum(avgpool_acc_per_image.values())) / len(avgpool_acc_per_image)
    gtprediction_final_acc = float(sum(gtprediction_acc_per_image.values())) / len(gtprediction_acc_per_image)
    lstm_final_acc = float(sum(lstm_acc_per_image.values())) / len(lstm_acc_per_image)

    if args.compute_mAP:
        avgpool_mean_ap = aggredate_video_map(avgpool_softmaxes_dic, labels_dic)
        lstm_mean_ap = aggredate_video_map(lstm_softmaxes_dic, labels_dic)
        gtprediction_mean_ap = aggredate_video_map(gtprediction_softmaxes_dic, labels_dic)

    if epoch_logger is not None:
        epoch_logger.info("Final accuracy from avgpooling at epoch {}: {}".format(epoch, avgpool_final_acc))
        epoch_logger.info("Final accuracy from lstm  at epoch {}: {}".format(epoch, lstm_final_acc))
        epoch_logger.info("Final accuracy from gt predictions at epoch {}: {}\n".format(epoch, gtprediction_final_acc))
        if args.compute_mAP:
            epoch_logger.info("mAP from avgpooling at epoch {}: {}".format(epoch, avgpool_mean_ap))
            epoch_logger.info("mAP from lstm at epoch {}: {}".format(epoch, lstm_mean_ap))
            epoch_logger.info("mAP from gt predictions at epoch {}: {}\n".format(epoch, gtprediction_mean_ap))

    if writer is not None:
        writer.add_text("test/log", "Avgpool final acc: {}".format(avgpool_final_acc), epoch)
        writer.add_text("test/log", "LSTM final acc: {}".format(lstm_final_acc), epoch)
        writer.add_text("test/log", "GT prediction final acc: {}".format(gtprediction_final_acc), epoch)
        writer.add_scalar("avgpool_accuracy/test", avgpool_final_acc, epoch)
        writer.add_scalar("lstm_accuracy/test", lstm_final_acc, epoch)
        writer.add_scalar("gtprediction_accuracy/test", gtprediction_final_acc, epoch)
        writer.add_scalar('avgpool_ce_loss/iter', avgpool_ce_losses.avg, epoch)
        writer.add_scalar('lstm_ce_loss/iter', lstm_ce_losses.avg, epoch)
        writer.add_scalar("loss/test", losses.avg, epoch)
    return avgpool_mean_ap, avgpool_final_acc, losses.avg, gtprediction_mean_ap

def main(args):
    logger = setup_logger(
        "Listen_to_look, classification",
        args.checkpoint_path,
        True
    )
    logger.debug(args)

    writer = None
    if args.visualization:
        writer = setup_tbx(
            args.checkpoint_path,
            True
        )
    if writer is not None:
        logger.info("Allowed Tensorboard writer")

    # create model
    builder = ModelBuilder()
    net_classifier = builder.build_classifierNet(args.embedding_size, args.num_classes).cuda()
    net_imageAudioClassify = builder.build_imageAudioClassifierNet(net_classifier, args).cuda()
    model = builder.build_audioPreviewLSTM(net_classifier, args)

    # define loss function (criterion) and optimizer
    criterion = {}
    criterion['CrossEntropyLoss'] = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    checkpointer = Checkpointer(model)
    
    if args.pretrained_model is not None:
        if not os.path.isfile(args.pretrained_model): 
            list_of_models = glob.glob(os.path.join(args.pretrained_model, "*.pth"))
            args.pretrained_model = max(list_of_models, key=os.path.getctime)
        logger.debug("Loading model only at: {}".format(args.pretrained_model))
        checkpointer.load_model_only(f=args.pretrained_model)

    model = torch.nn.parallel.DataParallel(model).cuda()
    
    # DATA LOADING
    val_ds, val_collate = create_validation_dataset(args,logger=logger)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.decode_threads,
        collate_fn=val_collate
    )
    
    video_mean_ap, video_acc, loss_avg, gtprediction_mean_ap = validate(args, 117, val_loader, model, criterion, val_ds=val_ds)
    print(
        "Testing Summary for checkpoint: {}\n"
        "video accuracy/mAP/gt mAP: {} \t {} \t {}\n".format(args.pretrained_model, video_acc*100, video_mean_ap*100, gtprediction_mean_ap*100)
    )

if __name__ == '__main__':
    args = get_parameters("Listen to Look Validataion")
    if args.pretrained_model is None:
        print("No model for validation - failing!!!")
        exit(0)
    main(args)
