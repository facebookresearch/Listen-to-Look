#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Part of the codes are borrowed or adapted from Bruno Korbar

import sys
import os
import logging


def setup_tbx(save_dir, is_master):
    from tensorboardX import SummaryWriter

    if not is_master:
        return None

    writer = SummaryWriter(save_dir)
    return writer


def setup_logger(name, save_dir, is_master, logname="run.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if not is_master:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = MyFormatter()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, logname))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# Custom formatter
class MyFormatter(logging.Formatter):

    err_fmt = "%(asctime)s %(name)s %(module)s: %(lineno)d: %(levelname)s: %(msg)s"
    dbg_fmt = "%(asctime)s %(module)s: %(lineno)d: %(levelname)s:: %(msg)s"
    info_fmt = "%(msg)s"

    def __init__(self):
        super().__init__(fmt="%(asctime)s %(name)s %(levelname)s: %(message)s",
                         datefmt=None,
                         style='%')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = MyFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._style._fmt = MyFormatter.err_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result
