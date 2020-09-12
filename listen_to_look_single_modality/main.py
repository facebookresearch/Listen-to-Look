#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Part of the codes are borrowed or adapted from Bruno Korbar

import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from models.models import ModelBuilder
from models.imageAudioClassify_model import ImageAudioClassifyModel

from opts import get_parameters
from validate import validate
from train import train_epoch

from data import create_training_dataset, create_validation_dataset

from utils.logging import setup_logger, setup_tbx
from utils.checkpointer import Checkpointer
from utils.scheduler import default_lr_scheduler

def main(args):

    os.makedirs(args.checkpoint_path, exist_ok=True)
    # Setup logging system
    logger = setup_logger(
        "Listen_to_look, audio_preview classification single modality",
        args.checkpoint_path,
        True
    )
    logger.debug(args)
    # Epoch logging
    epoch_log = setup_logger(
        "Listen_to_look: results",
        args.checkpoint_path, True,
        logname="epoch.log"
    )
    epoch_log.info("epoch,loss,acc,lr")

    writer = None
    if args.visualization:
        writer = setup_tbx(
            args.checkpoint_path,
            True
        )
    if writer is not None:
        logger.info("Allowed Tensorboard writer")

    # Define the model
    builder = ModelBuilder()
    net_classifier = builder.build_classifierNet(args.embedding_size, args.num_classes).cuda()
    net_imageAudioClassify = builder.build_imageAudioClassifierNet(net_classifier, args).cuda()
    model = builder.build_audioPreviewLSTM(net_classifier, args)
    model = model.cuda()
    
    # DATA LOADING
    train_ds, train_collate = create_training_dataset(args,logger=logger)
    val_ds, val_collate = create_validation_dataset(args,logger=logger)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.decode_threads,
        collate_fn=train_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=val_collate
    )

    args.iters_per_epoch = len(train_loader)
    args.warmup_iters = args.warmup_epochs * args.iters_per_epoch
    args.milestones = [args.iters_per_epoch * m for m in args.milestones]

    # define loss function (criterion) and optimizer
    criterion = {}
    criterion['CrossEntropyLoss'] = nn.CrossEntropyLoss().cuda()

    if args.freeze_imageAudioNet:
        param_groups = [{'params': model.queryfeature_mlp.parameters(), 'lr': args.lr},
                        {'params': model.prediction_fc.parameters(), 'lr': args.lr},
                        {'params': model.key_conv1x1.parameters(), 'lr': args.lr},
                        {'params': model.rnn.parameters(), 'lr': args.lr},
                        {'params': net_classifier.parameters(), 'lr': args.lr}]
        optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=1)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones)
    # make optimizer scheduler
    if args.scheduler:
        scheduler = default_lr_scheduler(optimizer, args.milestones, args.warmup_iters)

    cudnn.benchmark = True

    # setting up the checkpointing system
    write_here = True
    checkpointer = Checkpointer(model, optimizer, save_dir=args.checkpoint_path,
                                save_to_disk=write_here, scheduler=scheduler,
                                logger=logger)

    if args.pretrained_model is not None:
        logger.debug("Loading model only at: {}".format(args.pretrained_model))
        checkpointer.load_model_only(f=args.pretrained_model)

    if checkpointer.has_checkpoint():
        # call load checkpoint
        logger.debug("Loading last checkpoint")
        checkpointer.load()

    model = torch.nn.parallel.DataParallel(model).cuda()
    logger.debug(model)

    # Log all info
    if writer:
        writer.add_text("namespace", repr(args))
        writer.add_text("model", str(model))

    #
    # TRAINING
    #
    logger.debug("Entering the training loop")
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_accuracy, train_loss = train_epoch(args, epoch, train_loader, model,
                    criterion, optimizer,
                    scheduler,
                    logger, epoch_logger=epoch_log, checkpointer=checkpointer, writer=writer)

        test_map, test_accuracy, test_loss, _ = validate(args, epoch, val_loader, model, criterion,
                 epoch_logger=epoch_log, writer=writer)
        if writer is not None:
            writer.add_scalars('training_curves/accuracies', {'train': train_accuracy, 'val':test_accuracy}, epoch)
            writer.add_scalars('training_curves/loss', {'train': train_loss, 'val':test_loss}, epoch)
    
if __name__ == '__main__':
    args = get_parameters("Listen to Look")
    main(args)
