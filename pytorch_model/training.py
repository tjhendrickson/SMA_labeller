#!/usr/bin/env python3

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from model import UNet3D, CustomMRIImageDataset, DiceLoss
from utils import enumerateWithEstimate
import logging
import argparse
import datetime
import sys
import os

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class SMALabellerApp:
    def __init__(self):

        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--num_workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--batch_size',
                            help='Batch size to use for training',
                            default=2,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )
        parser.add_argument('--training_images',
                            help='Folder storing the training images',
                            default="/panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/pytorch_model/Dataset503_MICCAI2012_and_316_CBIT_SMA_GM_ribbon/imagesTr_resampled"
                            )
        parser.add_argument('--training_labels',
                            help='Folder storing the training labels',
                            default="/panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/pytorch_model/Dataset503_MICCAI2012_and_316_CBIT_SMA_GM_ribbon/labelsTr_resampled"
                            )
        parser.add_argument('--tb_prefix',
                            default='SMALabeller',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )

        #parser.add_argument('comment',
        #                    help="Comment suffix for Tensorboard run.",
        #                    nargs='?',
        #                    default='cconelea',
        #                    )
        # parse arguments
        cli_args = parser.parse_args()
        self.num_workers = cli_args.num_workers
        self.batch_size = cli_args.batch_size
        self.epochs = cli_args.epochs
        self.training_images = cli_args.training_images
        self.training_labels = cli_args.training_labels
        self.tb_prefix = cli_args.tb_prefix
        #self.comment = cli_args.comment

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        #self.trn_writer = None
        #self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        self.train_loader = DataLoader(CustomMRIImageDataset(annotations_dir=self.training_labels, img_dir=self.training_images),
                                       batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers, pin_memory=True)


    def initModel(self):
        model = UNet3D(n_class=1).to(self.device)
        log.info("Using '{}".format(self.device))
        return model

    def initOptimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
                                                        epochs=self.epochs,steps_per_epoch=len(self.train_loader))
        return optimizer,scheduler

    def computeBatchLoss(self,batch_idx,batch_tup,batch_size):
        input_t,label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        loss = DiceLoss()(input_g, label_g)

        # TODO keep going
        return loss.mean()

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def doTraining(self):
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.epochs):
            """
            log.info("Epoch {} of {}, batches of size {}".format(
                epoch,
                self.epochs,
                len(self.train_loader),
                self.batch_size
            ))
            """
            print("Epoch {} of {}, batches of size {}".format(
                epoch,
                self.epochs,
                len(self.train_loader),
                self.batch_size
            ))
            # TODO need to ascertain how this works
            batch_iter = enumerateWithEstimate(
                self.train_loader,
                "E{} Training".format(epoch),
                start_ndx=self.train_loader.num_workers,
            )
            for image, mask in self.train_loader:
                image, mask = image.cuda(), mask.cuda()

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    output = self.model(image)
                    loss = DiceLoss()(output, mask)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            self.scheduler.step()

    def main(self):
        self.model = self.initModel()
        self.optimizer, self.scheduler = self.initOptimizer()

        self.doTraining()


SMALabellerApp().main()












