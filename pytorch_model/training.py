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
import torch.nn as nn

from model import UNet3D, Custom3DMRIImageDataset, Custom3DMRIDatasetMONAI, DiceLoss,UNet2D, MONAIDiceLoss
from utils import enumerateWithEstimate,benchmark_loss_step
import logging
import argparse
import datetime
import sys
import os
from glob import glob

from monai.data import DataLoader as MONAIDataLoader
from monai.data import pad_list_data_collate

from sklearn.model_selection import train_test_split

import pdb

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

home_dir = os.path.expanduser('~')


def forward_slices_only(model, batch, device, slice_batch: int = 16):
    # unpack
    if isinstance(batch, dict):
        x = batch['image'].as_tensor().to(device, non_blocking=True)
    else:
        x, _ = batch
    x = x.to(device, non_blocking=True)

    # [B,C,H,W,D] -> [B*D,C,H,W]
    B, C, H, W, D = x.shape
    x = x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)

    # micro-batch so calibration never OOMs
    N = x.shape[0]
    for s in range(0, N, slice_batch):
        e = min(s + slice_batch, N)
        _ = model(x[s:e])  # forward only
    


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
        parser.add_argument('--val_set_size',
                            help='Size of the validation set as a fraction of the training set',
                            default=0.2,
                            type=float,
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
        self.val_set_size = cli_args.val_set_size
        self.comment = cli_args.comment

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None


        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        self.data_dicts = self.build_data_dicts()

        train_files, val_files = train_test_split(self.data_dicts, test_size=self.val_set_size, random_state=42)

        self.train_loader = MONAIDataLoader(Custom3DMRIDatasetMONAI(train_files),
            batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,
            pin_memory=True,collate_fn=pad_list_data_collate)

        self.val_loader = MONAIDataLoader(Custom3DMRIDatasetMONAI(val_files),
            batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,
            pin_memory=True,collate_fn=pad_list_data_collate)
        log.info("Total training samples: {}".format(len(self.train_loader)))
        log.info("Total validation samples: {}".format(len(self.val_loader)))

        self.dice_loss = DiceLoss()

    def build_data_dicts(self):
        label_paths = sorted(glob(os.path.join(self.training_labels, '*.nii.gz')))
        data_dicts = []
        for label_path in label_paths:
            base = os.path.basename(label_path).split('.')[0]
            image_path = os.path.join(self.training_images, base + '_0000.nii.gz')
            if os.path.exists(image_path):
                data_dicts.append({
                    "image": image_path,
                    "label": label_path
                })
        return data_dicts

    def reset_bn_running_stats(self):
        """
        Reset the running statistics of all BatchNorm2d layers in the model.
        This is useful for re-initializing the model's BatchNorm layers
        after loading a new dataset or changing the model architecture.
        """
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()  # zero mean, one var, reset counters

    @torch.no_grad()
    def calibrate_bn(self, forward_fn, warmup_batches: int = 8):
        """
        Put BN layers into train() so they update running stats.
        Do a few forward passes (no loss/no backprop) to refresh stats.
        Keep it FP32 for numerics; micro-batch inside forward_fn to avoid OOM.
        """
        self.model.train()
        # (Optional) disable autocast to avoid fp16 noise in BN stats
        with torch.cuda.amp.autocast(enabled=False), torch.inference_mode():
            for i, batch in enumerate(self.train_loader):
                if i >= warmup_batches:
                    break
                forward_fn(batch)  # forward pass only (no loss/backward)
        # After calibration you'll switch to eval() before validation

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
        if isinstance(batch, dict):
            # MONAI DataLoader returns a dict
            input_t = batch['image']
            label_t = batch['label']
        elif isinstance(batch, list):
            # Custom DataLoader returns a list
            input_t, label_t = batch
        
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        
        # Reshape from [B, C, H, W, D]  [B×D, C, H, W]
        B, C, H, W, D = input_g.shape
        input_slices = input_g.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
        label_slices = label_g.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)

        # Forward pass
        with torch.cuda.amp.autocast():
            output_g = self.model(input_slices)

        if torch.isnan(output_g).any():
            print(f"[NaN Warning] NaNs in model output at batch {batch_idx}")
        if torch.isnan(label_slices).any():
            print(f"[NaN Warning] NaNs in label at batch {batch_idx}")

        # Compute loss
        
        #loss = self.dice_loss(output_g, label_slices)
        with torch.cuda.amp.autocast(enabled=False): # Loss in FP32
            loss = MONAIDiceLoss(output_g.float(),label_slices.float())

        return loss # return loss for entire batch
    
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
            print("Epoch {} of {}, batches of size {}, total batch {}".format(
                epoch,
                self.epochs,
                self.batch_size,
                len(self.train_loader)
            ))

            for batch_idx, batch_data in enumerate(self.train_loader):
                
                # Use autocast for mixed precision training
                with torch.cuda.amp.autocast():
                    loss = self.compute3dBatchLoss(batch_idx, batch_data, self.batch_size)
                
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
            print("Epoch {} of {}, batches of size {}, epoch size {}".format(
                epoch,
                self.epochs,
                self.batch_size,
                len(self.train_loader)
            ))
            training_loss = 0.0
            for batch_idx, batch_data in enumerate(self.train_loader):
                                

                loss, elapsed, peak_mem = benchmark_loss_step(self.compute2dBatchLoss, batch_idx, batch_data, self.batch_size)
                # Log the loss
                print(f"[Epoch {epoch} | Batch {batch_idx}] Loss: {loss.item():.4f} | Time: {elapsed:.3f}s | Peak Mem: {peak_mem:.2f} MB")
                self.logMetrics(epoch, batch_idx, loss)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                training_loss += loss.item()
                self.model.zero_grad()
                
            avg_training_loss = training_loss / len(self.train_loader)
            print(f"Epoch {epoch} Average Training Loss: {avg_training_loss:.4f}")
            
            # ------------------
            # BN CALIBRATION
            # ------------------
            self.reset_bn_running_stats()  # (do this each epoch, right before validation)
            self.calibrate_bn(forward_fn=lambda b: forward_slices_only(self.model, b, self.device, slice_batch=16),  # tune slice_batch if needed
                warmup_batches=8,  # 4-16 is typical
            )

            # Validation step
            
            self.model.eval()
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(self.val_loader):

                    loss = self.compute2dBatchLoss(batch_idx, batch_data, self.batch_size)
                    
                    # Log the validation loss
                    
                    print("Validation - Epoch {} Batch {} Loss: {:.4f}".format(
                        epoch, batch_idx, loss.item()))
                    if self.val_writer is not None:
                        self.val_writer.add_scalar('loss', loss.item(), epoch * len(self.val_loader) + batch_idx)
                        self.val_writer.flush()
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(self.val_loader)
            print(f"Epoch {epoch} Average Validation Loss: {avg_val_loss:.4f}")

                        
            
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












