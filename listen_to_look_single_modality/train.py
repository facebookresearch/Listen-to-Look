#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Part of the codes are borrowed or adapted from Bruno Korbar

import time
import torch
from utils.metrics import AverageMeter, calculate_accuracy
from utils.utils import init_hidden
import torch.nn.functional as F

def train_epoch(args, epoch, data_loader, model, criterion, optimizer, scheduler=None,
                logger=None, epoch_logger=None, checkpointer=None, writer=None):

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avgpool_ce_losses = AverageMeter()
    lstm_ce_losses = AverageMeter()
    losses = AverageMeter()
    avgpool_accuracies = AverageMeter()
    lstm_accuracies = AverageMeter()
    gtprediction_accuracies = AverageMeter()

    end_time = time.time()
    for i, data in enumerate(data_loader):
        i = i + 1
        gt_features, gt_predictions, feature_masks, labels, _ = data
        gt_features = gt_features.cuda()
        feature_masks = feature_masks.cuda()
        labels = labels.cuda()
        data_time.update(time.time() - end_time)
        batch_size = gt_features.shape[0]
        predictions, selected_gt_predictions, selected_step_predictions = model.forward(gt_features, gt_predictions, feature_masks, args.episode_length, args.gt_feature_eval)
        lstm_acc = calculate_accuracy(selected_step_predictions[-1], labels)
        gtprediction_acc = calculate_accuracy(selected_gt_predictions, labels)
        avgpool_acc = calculate_accuracy(predictions, labels)

        lstm_ce_loss = criterion['CrossEntropyLoss'](selected_step_predictions[-1], labels)
        avgpool_ce_loss = criterion['CrossEntropyLoss'](predictions, labels)

        #final loss to use
        loss = 0
        if args.with_avgpool_ce_loss:
            loss = loss + avgpool_ce_loss
        if args.with_lstm_ce_loss:
            loss = loss + lstm_ce_loss

        avgpool_ce_losses.update(avgpool_ce_loss.data.item(), batch_size)
        lstm_ce_losses.update(lstm_ce_loss.data.item(), batch_size)
        losses.update(loss.data.item(), batch_size)
        avgpool_accuracies.update(avgpool_acc, batch_size)
        gtprediction_accuracies.update(gtprediction_acc, batch_size)
        lstm_accuracies.update(lstm_acc, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if writer is not None:
            step = i + (epoch * len(data_loader))
            writer.add_scalar('avgpool_accuracy/iter', avgpool_accuracies.avg, step)
            writer.add_scalar('lstm_accuracy/iter', lstm_accuracies.avg, step)
            writer.add_scalar('gtprediction_accuracy/iter', gtprediction_accuracies.avg, step)
            writer.add_scalar('avgpool_ce_loss/iter', avgpool_ce_losses.avg, step)
            writer.add_scalar('lstm_ce_loss/iter', lstm_ce_losses.avg, step)
            writer.add_scalar('loss/iter', losses.avg, step)
            writer.add_scalar('data_time/iter', data_time.val, step)
            writer.add_scalar('lr/iter', optimizer.param_groups[0]['lr'], step)

        if i % args.print_freq == 0:
            msg = ('Epoch: [{0}][{1}/{2}]\t'
                'LR: {3}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Avgpool CE Loss {avgpool_ce_loss.val:.4f} ({avgpool_ce_loss.avg:.4f})\t'
                'LSTM CE Loss {lstm_ce_loss.val:.4f} ({lstm_ce_loss.avg:.4f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Avgpool Acc {avgpool_acc.val:.3f} ({avgpool_acc.avg:.3f})\t'
                'LSTM Acc {lstm_acc.val:.3f} ({lstm_acc.avg:.3f})\t'
                'GT Prediction Acc {gtprediction_acc.val:.3f} ({gtprediction_acc.avg:.3f})\n'.format(
                    epoch,
                    i,
                    len(data_loader),
                    optimizer.param_groups[0]['lr'],
                    batch_time=batch_time,
                    data_time=data_time,
                    avgpool_ce_loss=avgpool_ce_losses,
                    lstm_ce_loss=lstm_ce_losses,
                    loss=losses,
                    avgpool_acc=avgpool_accuracies,
                    lstm_acc=lstm_accuracies,
                    gtprediction_acc=gtprediction_accuracies))
            logger.info(msg)
            if writer is not None:
                writer.add_text('logs', msg, i * epoch)

        if scheduler is not None:
            scheduler.step()

    # TODO: do I want to save at iteration times?
    if epoch % args.checkpoint_freq == 0 and checkpointer is not None:
        checkpointer.save(epoch, "model_{:07d}".format(epoch), args)
    if epoch == args.epochs - 1 and checkpointer is not None:
        checkpointer.save(epoch, "model_final", args)

    if writer is not None:
        writer.add_scalar('avgpool_ce_loss/epoch', avgpool_ce_losses.avg, epoch)
        writer.add_scalar('lstm_ce_loss/epoch', lstm_ce_losses.avg, epoch)
        writer.add_scalar('loss/epoch', losses.avg, epoch)
        writer.add_scalar('lr/epoch', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('data_time/epoch', data_time.avg, epoch)
    # log the info about the epoch
    if epoch_logger is not None:
        epoch_logger.info(
            "{}, {}, {}".format(
                epoch,
                losses.avg,
                optimizer.param_groups[0]['lr']
            ))
    return lstm_accuracies.avg, losses.avg