#!/usr/bin/env python3

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from model import UNet3D, Custom3DMRIImageDataset, Custom3DMRIDatasetMONAI, DiceLoss,UNet2D, MONAIDiceLoss
from utils import enumerateWithEstimate
import logging
import argparse
import datetime
import sys
import os

from monai.data import DataLoader as MONAIDataLoader
from monai.data import pad_list_data_collate

import pdb

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

home_dir = os.path.expanduser('~')

class SMALabellerApp:
    def __init__(self):

        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--unet_dimensions',
                            help='Dimensions of the UNet model to use. 2D or 3D',
                            default='3D',
                            choices=['2D', '3D'],
                            type=str)
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

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='cconelea',
                            )
        # parse arguments
        cli_args = parser.parse_args()
        self.unet_dimensions = cli_args.unet_dimensions
        self.num_workers = cli_args.num_workers
        self.batch_size = cli_args.batch_size
        self.epochs = cli_args.epochs
        self.training_images = cli_args.training_images
        self.training_labels = cli_args.training_labels
        self.tb_prefix = cli_args.tb_prefix
        self.comment = cli_args.comment

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        #self.train_loader = DataLoader(Custom3DMRIImageDataset(img_dir=self.training_images, annotations_dir=self.training_labels),
        #                               batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers, pin_memory=True)
        self.train_loader = MONAIDataLoader(Custom3DMRIDatasetMONAI(img_dir=self.training_images,label_dir=self.training_labels),
                                       batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,
                                       collate_fn=pad_list_data_collate)
        self.totalTrainingSamples_count = len(self.train_loader.dataset)
        log.info("Total training samples: {}".format(self.totalTrainingSamples_count))

        self.dice_loss = DiceLoss()

    def init3dModel(self):
        model = UNet3D(n_class=1).to(self.device)
        log.info("Using '{}".format(self.device))
        return model
    
    def init2dModel(self):
        model = UNet2D(n_class=1).to(self.device)
        log.info("Using '{}".format(self.device))
        return model

    def initOptimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
                                                        epochs=self.epochs,steps_per_epoch=len(self.train_loader))
        return optimizer,scheduler

    def compute2dBatchLoss(self,batch_idx,batch,batch_size):
        running_loss = 0.0
        if isinstance(batch, dict):
            # MONAI DataLoader returns a dict
            input_t = batch['image']
            label_t = batch['label']
        elif isinstance(batch, list):
            # Custom DataLoader returns a list
            input_t, label_t = batch

        input_g = input_t.to(self.device, non_blocking=True)
        if batch_size == 1:
            for i in range(input_g.shape[-1]): # Iterate over the depth dimension
                output_g = self.model(input_g[:, :, :, :, i]) # don't want i:i+1 here as it is a 2D slice
                label_g = label_t.to(self.device, non_blocking=True)[:, :, :, :, i]
                #loss = self.dice_loss(output_g, label_g)
                loss = MONAIDiceLoss(output_g, label_g)
                running_loss += loss
        elif batch_size > 1:
            for i in range(input_g.shape[0]): # Iterate over the batch dimension
                for j in range(input_g.shape[-1]): # Iterate over the depth dimension
                    output_g = self.model(input_g[i:i+1, :, :, :, j])
                    label_g = label_t.to(self.device, non_blocking=True)[i:i+1, :, :, :, j]
                    #loss = self.dice_loss(output_g, label_g)
                    loss = MONAIDiceLoss(output_g, label_g)
                    running_loss += loss

        return running_loss / (input_g.shape[0] * input_g.shape[-1])  # Average loss over batch and depth
    
    def compute3dBatchLoss(self,batch_idx,batch_tup,batch_size):

        input_t, label_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        output_g = self.model(input_g)
        label_g = label_t.to(self.device, non_blocking=True)

        loss = self.dice_loss(output_g, label_g)
        return loss

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join(home_dir,'runs', self.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.comment)
    
    def logMetrics(self,epoch, batch_idx, loss):
        if self.trn_writer is not None:
            self.trn_writer.add_scalar('loss', loss.item(), epoch * len(self.train_loader) + batch_idx)
            self.trn_writer.flush()
        else:
            log.warning("Tensorboard writer not initialized. Metrics will not be logged.")

    def do3dTraining(self):
        scaler = torch.cuda.amp.GradScaler()
        self.model.zero_grad()
        self.model.train()
        for epoch in range(self.epochs):
            print("Epoch {} of {}, batches of size {}".format(
                epoch,
                self.epochs,
                len(self.train_loader),
                self.batch_size
            ))
            batch_iter = enumerateWithEstimate(
                self.train_loader,
                "E{} Training".format(epoch),
                start_ndx=self.train_loader.num_workers,
            )
            for batch_idx, batch_tup in batch_iter:
                
                # Use autocast for mixed precision training
                with torch.cuda.amp.autocast():
                    loss = self.compute3dBatchLoss(batch_idx, batch_tup, self.batch_size)
                
                # Log the loss
                if batch_idx % 10 == 0:
                    print("Epoch {} Batch {} Loss: {:.4f}".format(
                        epoch, batch_idx, loss.item()))
                    self.logMetrics(epoch, batch_idx, loss)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                self.model.zero_grad()

    def do2dTraining(self):
        scaler = torch.cuda.amp.GradScaler()
        self.model.zero_grad()
        self.model.train()
        for epoch in range(self.epochs):
            print("Epoch {} of {}, batches of size {}".format(
                epoch,
                self.epochs,
                len(self.train_loader),
                self.batch_size
            ))
            batch_iter = enumerateWithEstimate(
                self.train_loader,
                "E{} Training".format(epoch),
                start_ndx=self.train_loader.num_workers,
            )
            
            for batch_idx, batch_tup in batch_iter:
                
                # Use autocast for mixed precision training
                with torch.cuda.amp.autocast():
                    loss = self.compute2dBatchLoss(batch_idx, batch_tup, self.batch_size)
                
                # Log the loss
                if batch_idx % 10 == 0:
                    print("Epoch {} Batch {} Loss: {:.4f}".format(
                        epoch, batch_idx, loss.item()))
                    self.logMetrics(epoch, batch_idx, loss)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                self.model.zero_grad()
            
    def main(self):
        log.info("Starting SMALabellerApp with unet_dimensions: {}".format(self.unet_dimensions))
        if self.unet_dimensions == '2D':
            self.model = self.init2dModel()
            self.optimizer, self.scheduler = self.initOptimizer()
            self.initTensorboardWriters()
            self.do2dTraining()
        elif self.unet_dimensions == '3D':
            self.model = self.init3dModel()
            self.optimizer, self.scheduler = self.initOptimizer()
            self.initTensorboardWriters()
            self.do3dTraining()

SMALabellerApp().main()












