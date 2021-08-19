# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
Code adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
'''

import os
import sys
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter

import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class Visualizer():
    def __init__(self, top_out_path, use_wandb=False):  # This will cause error in the very old train scripts
        self.writer = SummaryWriter(top_out_path)
        self.use_wandb = use_wandb

    # |visuals|: dictionary of images to save
    def log_images(self, visuals, step):
        for label, image_numpy in visuals.items():
            self.writer.add_images(
                label, [image_numpy], step)

    # scalars: dictionary of scalar labels and values
    def log_scalars(self, scalars, step, main_tag='metrics'):
        self.writer.add_scalars(main_tag=main_tag, tag_scalar_dict=scalars, global_step=step)   # hook not working...
        if self.use_wandb:
            wandb.log(scalars, step=step)
