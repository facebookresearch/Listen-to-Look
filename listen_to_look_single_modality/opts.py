#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Part of the codes are borrowed or adapted from Bruno Korbar

import argparse

def get_parameters(description="Generic parser for deep learnining"):
    """
    Generic argparse parameter getter for deep learning
    :return: argparse object with training parameters
    """
    parser = argparse.ArgumentParser(
        description=description)

    # Dataset API
    parser.add_argument(
        '--num_classes',
        default=400,
        type=int,
        help='Number of classes'
    )
    parser.add_argument(
        "--train_dataset_file",
        type=str,
        default=None,
        help="Path to the pickle dataset file for training",
    )
    parser.add_argument(
        "--test_dataset_file",
        type=str,
        default=None,
        help="Path to the pickle dataset file for validation/testing",
    )

    # Preprocessing
    parser.add_argument(
        '--normalization',
        default=False,
        action='store_true',
        help="Should we use input normalization?"
    )
    # Model parameters
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="number of hidden units for lstm"
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=10,
        help="number of step to perform adaptive skim for lstm"
    )
    parser.add_argument(
        '--with_avgpool_ce_loss',
        action='store_true',
        default=False,
        help='classification loss on avgpooled feature'
    )
    parser.add_argument(
        '--with_lstm_ce_loss',
        action='store_true',
        default=False,
        help='ce loss on last step of lstm prediction'
    )
    parser.add_argument(
        '--with_replacement',
        action='store_true',
        default=False,
        help='whether to enforce always selecting different frames'
    )    
    parser.add_argument(
        '--imageKey',
        action='store_true',
        default=False,
        help='whether using image features as keys, otherwise using audio features'
    )
    parser.add_argument(
        '--gt_feature_eval',
        action='store_true',
        default=False,
        help='whether using gt feature during eval'
    )   
    parser.add_argument(
        '--mean_feature_as_start',
        action='store_true',
        default=False,
        help='whether to use mean feature as the start feature'
    )   

    # Using --pretrained model initializes the model weights without
    # updating the optimizer; using the checkpoint path updates
    # model, optimizer and scheduler
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=512,
        help="embedding size"
    )
    parser.add_argument(
        "--weights_audioImageModel",
        type=str,
        default='',
        help="Model weights for audioImage stream"
    )
    parser.add_argument(
        "--weights_classifier",
        type=str,
        default='',
        help="weights for fc layer of video model"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Pretrained model state to use"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=3,
        help="How often to save the checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=".",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--subsample_factor",
        type=int,
        default=2,
        help="subsampling factor of feature banks"
    )
    parser.add_argument(
        '--feature_interpolate',
        default=False,
        action='store_true',
        help='whether to interpolate features during training'
    )

    # Training parameters
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Starting epoch in case of continuation [0]"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of epochs to run must be > args.start_epoch"
    )
    parser.add_argument(
        '--milestones',
        nargs='+',
        help='Reduce learning rate',
        default=[20, 40],
        type=int
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Base learnining rate for training"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum training parameter"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay training parameter"
    )
    parser.add_argument(
        '--scheduler',
        default=False,
        action='store_true',
        help='Turns on warmup scheduler'
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of epochs used for distributed warmup; is transformed to iters for linearity later "
    )
    parser.add_argument(
        '--freeze_imageAudioNet',
        action='store_true',
        default=False,
        help='whether freeze the imageAudioNet weights'
    )   

    # technical dataloader parameters
    parser.add_argument(
        '--decode_threads',
        default=24,
        type=int,
        help='Number of worker threads for dataloading'
    )

    # Logging
    parser.add_argument(
        '--visualization',
        action='store_true',
        help='Setup the tensorboard logger'
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default="20",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--compute_mAP",
        default=False,
        action='store_true',
        help='Whether compute and log mAP'
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = get_parameters()
    print(args)
