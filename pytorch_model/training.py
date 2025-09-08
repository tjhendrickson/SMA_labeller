#!/usr/bin/env python3

from ray import train

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from model.model import UNet3D, Custom3DMRIImageDataset, Custom3DMRIDatasetMONAI, UNet2D, MONAIDiceLoss,torch_BCEWithLogitsLoss,MONAIDiceCELoss,EarlyStopping
from utils.utils import enumerateWithEstimate,benchmark_loss_step

import logging

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
    """ Forward pass through the model using slices of the input batch.
    This function reshapes the input tensor to process slices in a micro-batch manner,
    which helps in avoiding out-of-memory (OOM) errors during calibration.
    Args:
        model (torch.nn.Module): The model to run the forward pass.
        batch (tuple or dict): Input batch containing the image tensor.
        device (torch.device): The device to run the model on (CPU or GPU).
        slice_batch (int): Number of slices to process in each micro-batch.
    """
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
    def __init__(self,args=None):
        import argparse

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
        
        if isinstance(args, dict): # if dictionary, convert to CLI args
            cli_args = parser.parse_args(self._dict_to_cli(args))
        elif isinstance(args, list):
            cli_args = parser.parse_args(args)
        else:
            cli_args, _ = parser.parse_known_args()

        # set attributes from CLI args
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

    def _dict_to_cli(self, arg_dict):
        """Convert dict to CLI-style args: {'batch_size': 4}  ['--batch_size', '4']"""
        cli_args = []
        for k, v in arg_dict.items():
            if isinstance(v, bool):
                if v:
                    cli_args.append(f"--{k}")
            else:
                cli_args.extend([f"--{k}", str(v)])
        return cli_args

    def _set_device(self):
        """
        Set the device for training based on CUDA availability.
        This method is called before any CUDA operations.
        """
        if torch.cuda.is_available():
            # Ray makes a single visible GPU per trial; it's "0" within the trial
            torch.cuda.set_device(torch.cuda.current_device())
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def _grad_global_norm(self, max_params=100000000):
        """
        Compute the global norm of gradients across all model parameters.
        This is useful for gradient clipping or logging.
        Returns:
            float: Global norm of gradients.
        """
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += (p.grad.detach().data.float().norm(2) ** 2).item()
        return total ** 0.5
    
    def _current_lr(self):
        """
        Get the current learning rate from the optimizer.
        This is useful for logging the learning rate during training.
        Returns:
            float: Current learning rate.
        """
        # OneCycleLR can have param groups; log the first for simplicity
        return float(self.optimizer.param_groups[0]["lr"])
    
    def _tb_log_val_images(self, batch, logits, epoch, max_slices=3):
        """
        Log a few (img, label, pred) 2D slices to TensorBoard during validation.
        Expects:
        - batch['image'], batch['label']: [B, C, H, W, D] (MONAI MetaTensors)
        - logits: [B*D, 1, H, W] (produced after your 2D-slice reshaping)
        """
        if not self.val_writer:
            return

        try:
            with torch.no_grad():
                imgs = batch['image'].as_tensor().detach().cpu()   # [B, C, H, W, D]
                lbls = batch['label'].as_tensor().detach().cpu()   # [B, 1, H, W, D]

                B, C, H, W, D = imgs.shape
                take = min(max_slices, D)

                # Reassemble logits back to [B, D, 1, H, W] to index slices per volume
                logits = logits.detach().cpu()
                logits = logits.view(B, D, 1, H, W)  # same order as your flattening

                # Log only the volumes around %75 of the max D slices
                location = int(D * 0.5)
                for i in range(take):
                    img = imgs[0, 0, :, :, location+i].unsqueeze(0)               # [1, H, W]
                    lbl = lbls[0, 0, :, :, location+i].unsqueeze(0)               # [1, H, W]
                    pred = torch.sigmoid(logits[0, location+i, 0, :, :]).unsqueeze(0)  # [1, H, W]

                    self.val_writer.add_image(f"val/img/{i}",  img,  epoch)
                    self.val_writer.add_image(f"val/lbl/{i}",  lbl,  epoch)
                    self.val_writer.add_image(f"val/pred/{i}", pred, epoch)
        except Exception as e:
            log.warning(f"Image logging skipped: {e}")

    def _atomic_save(self,obj, path_tmp, path_final):
        """
        Atomically save `obj` to `path_final` by first writing to `path_tmp`
        and then renaming it. This prevents partial writes.
        """

        torch.save(obj, path_tmp)
        os.replace(path_tmp, path_final)  # atomic rename on POSIX

    def save_checkpoint(self,path,epoch,global_step,monitor_value,config):
        """
        Save a training checkpoint.
        Args:
            path (str): Path to save the checkpoint.
            model (torch.nn.Module): Model to save.
            optimizer (torch.optim.Optimizer): Optimizer to save.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to save.
            scaler (torch.cuda.amp.GradScaler): GradScaler to save.
            epoch (int): Current epoch number.
            global_step (int): Current global step number.
            monitor_value (float): Value of the monitored metric for early stopping.
            config (dict): Configuration dictionary.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "monitor_value": float(monitor_value) if monitor_value is not None else None,
            "config": config,
            # Optional RNGs:
            # "torch_rng": torch.get_rng_state(),
            # "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        self._atomic_save(payload, path + ".tmp", path)

    def cleanup_old(self,pattern, keep=3):
        """
        Remove old files matching `pattern`, keeping only the newest `keep` files.
        Args:
            pattern (str): Glob pattern to match files.
            keep (int): Number of newest files to keep.
        """
        # keep newest `keep` files matching pattern
        import glob
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for f in files[keep:]:
            try: os.remove(f)
            except: pass

    def build_data_dicts(self):
        """
        Build a list of dictionaries containing paths to training images and labels.
        Each dictionary contains:
            - "image": Path to the training image file.
            - "label": Path to the corresponding label file.
        Returns:
            data_dicts (list): List of dictionaries with image and label paths.
        """
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
        """
        Initialize the 3D UNet model.
        Returns:
            model: The initialized UNet3D model.
        """
        model = UNet3D(n_class=1).to(self.device)
        log.info("Using '{}".format(self.device))
        return model
    
    def init2dModel(self):
        """
        Initialize the 2D UNet model.
        Returns:
            model: The initialized UNet2D model.
        """
        model = UNet2D(n_class=1).to(self.device)
        log.info("Using '{}".format(self.device))
        return model

    def initOptimizer(self,config):
        """
        Initialize the optimizer and learning rate scheduler.
        Returns:
            optimizer: The optimizer instance.
            scheduler: The learning rate scheduler instance.
        """
        lr = float(config["lr"])
        max_lr = float(config["max_lr"])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                                        epochs=self.epochs,steps_per_epoch=len(self.train_loader))
        return optimizer,scheduler

    def initTensorboardWriters(self,hparams: dict):
        """
        Initialize TensorBoard writers for training and validation.
        Args:
            hparams (dict): Hyperparameters to log in TensorBoard.
        """

        if self.trn_writer is None:
            self.log_dir = os.path.join(home_dir,'runs', self.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=self.log_dir + '-trn_cls-' + self.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.log_dir + '-val_cls-' + self.comment)
        
        if hparams:
            self.trn_writer.add_text("hparams/json", str(hparams))

    def log_train_batch(self, epoch, batch_idx, loss, elapsed):
        """
        Log training metrics for a single batch.
        Args:
            epoch (int): Current epoch number.
            batch_idx (int): Index of the current batch.
            loss (torch.Tensor): Loss value for the current batch.
            elapsed (float): Time taken for the current batch in seconds.
        """
        step = epoch * len(self.train_loader) + batch_idx
        if self.trn_writer:
            self.trn_writer.add_scalar("Loss/batch", loss.item(), step)
            self.trn_writer.add_scalar("LR/batch", self._current_lr(), step)
            self.trn_writer.add_scalar("Time/iter_sec", elapsed, step)
    
    def log_epoch_metrics(self, epoch, avg_train_loss, avg_val_loss=None):
        """
        Log metrics for the entire epoch.
        Args:
            epoch (int): Current epoch number.
            avg_train_loss (float): Average training loss for the epoch.
            avg_val_loss (float, optional): Average validation loss for the epoch. Defaults to None.
        """
        if self.trn_writer:
            self.trn_writer.add_scalar("Loss/epoch_train", avg_train_loss, epoch)
        if self.val_writer and avg_val_loss is not None:
            self.val_writer.add_scalar("Loss/epoch_val", avg_val_loss, epoch)

    def compute2dBatchLoss(self,batch_idx,batch,batch_size,config,epoch,validation_loop=False):
        """
        Compute the loss for a single batch in 2D training.
        Args:

            batch_idx (int): Index of the current batch.
            batch (tuple): A tuple containing input and label tensors.
            batch_size (int): Size of the batch.
            config (dict): Configuration dictionary containing loss function and other parameters.
            epoch (int): Current epoch number.
            validation_loop (bool): Whether this is during validation. Defaults to False.
        Returns:
            loss (torch.Tensor): Computed loss for the batch.
        """

        # Unpack the batch
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
        

        # Log validation images
        if batch_idx == 0 and validation_loop == True:  # log only the first batch
            self._tb_log_val_images(batch, output_g.float(), epoch, max_slices=3)

        if torch.isnan(output_g).any():
            print(f"[NaN Warning] NaNs in model output at batch {batch_idx}")
        if torch.isnan(label_slices).any():
            print(f"[NaN Warning] NaNs in label at batch {batch_idx}")

        # Compute loss
        with torch.cuda.amp.autocast(enabled=False): # Loss in FP32
            if config['loss_function'] == 'dice':
                loss = MONAIDiceLoss(output_g.float(),label_slices.float())
            elif config['loss_function'] == 'cross_entropy':
                loss = torch_BCEWithLogitsLoss(output_g.float(),label_slices.float())
            elif config['loss_function'] == 'combination':
                loss = MONAIDiceCELoss(output_g.float(),label_slices.float())

        return loss # return loss for entire batch
    """
    def compute3dBatchLoss(self,batch_idx,batch_tup,batch_size):

        input_t, label_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        output_g = self.model(input_g)
        label_g = label_t.to(self.device, non_blocking=True)

        loss = self.dice_loss(output_g, label_g)
        return loss
    """
    """
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
                    #self.log_train_batch(epoch, batch_idx, loss, elapsed)
                #self.log_epoch_metrics(epoch, avg_train_loss=avg_training_loss, avg_val_loss=average)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                self.model.zero_grad()
    """

    def do2dTraining(self,config,ray_tune=False):
        """
        Perform 2D training using the specified configuration.
        Args:
            config (dict): Configuration dictionary containing training parameters.
            ray_tune (bool): Whether to report metrics to Ray Tune. Defaults to False.
        """

        # --- set device per trial, before any CUDA ops ---
        self._set_device()

        self.data_dicts = self.build_data_dicts()
        
        train_files, val_files = train_test_split(self.data_dicts, test_size=self.val_set_size, random_state=42)

        self.train_loader = MONAIDataLoader(Custom3DMRIDatasetMONAI(train_files,augmentations=config['augmentations']),
            batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,
            pin_memory=True,collate_fn=pad_list_data_collate)

        self.val_loader = MONAIDataLoader(Custom3DMRIDatasetMONAI(val_files,augmentations=False),
            batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,
            pin_memory=True,collate_fn=pad_list_data_collate)

        self.model = self.init2dModel()


        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self.initOptimizer(config)
        # Initialize TensorBoard writers
        self.initTensorboardWriters(hparams=dict(
            **config,
            unet_dimensions=self.unet_dimensions,
            batch_size=self.batch_size,
            epochs=self.epochs,
            training_images=self.training_images,
            training_labels=self.training_labels,
            num_workers=self.num_workers
        ))

        # --- set up saving and logging parameters ---
        
        K_PERIOD = 5     # every 5 epochs
        KEEP_PERIODIC = 3
        best_val = None
        global_step = 0
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        best_path = os.path.join(ckpt_dir, 'best.pth')
        last_path = os.path.join(ckpt_dir, 'last.pth')
        periodic_tmpl = os.path.join(ckpt_dir, 'periodic_{epoch:04d}.pth')

        # log model graph
        try:
            dummy = torch.randn(1, 1, 128, 128).to(self.device)  # [B, C, H, W]
            self.trn_writer.add_graph(self.model, dummy)
        except Exception as e:
            log.warning(f"Model graph logging failed: {e}")

        ckpt_dir = os.path.join(home_dir, 'runs', self.tb_prefix, self.time_str, 'checkpoints')
        early_stopper = EarlyStopping(
            patience=15,         # tweak: 10-20 common
            min_delta=0.0,       # e.g., 0.001 to require a meaningful drop
            mode='min',
            restore_best=True,
            ckpt_dir=ckpt_dir
        )
        log.info("Total training samples: {}".format(len(self.train_loader)))
        log.info("Total validation samples: {}".format(len(self.val_loader)))

        self.scaler = torch.cuda.amp.GradScaler()
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
                                
                loss, elapsed, peak_mem = benchmark_loss_step(self.compute2dBatchLoss, batch_idx, batch_data, self.batch_size,config,epoch)
                # Log the loss
                print(f"[Epoch {epoch} | Batch {batch_idx}] Loss: {loss.item():.4f} | Time: {elapsed:.3f}s | Peak Mem: {peak_mem:.2f} MB")
                self.log_train_batch(epoch, batch_idx, loss, elapsed)
                
                # backward pass
                self.scaler.scale(loss).backward()

                # log gradient norm
                grad_norm = self._grad_global_norm()
                if self.trn_writer:
                    step = epoch * len(self.train_loader) + batch_idx
                    self.trn_writer.add_scalar("GradNorm/batch", grad_norm, step)
                # optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                training_loss += loss.item()
                self.model.zero_grad()
                global_step += 1
                
            avg_training_loss = training_loss / len(self.train_loader)
            print(f"Epoch {epoch} Average Training Loss: {avg_training_loss:.4f}")
            self.log_epoch_metrics(epoch, avg_train_loss=avg_training_loss, avg_val_loss=None)

            # histograms of parameters and gradients, once per epoch
            if self.trn_writer:
                for name, param in self.model.named_parameters():
                    self.trn_writer.add_histogram(f"Params/{name}", param.detach().cpu(), epoch)
                    if param.grad is not None:
                        self.trn_writer.add_histogram(f"Grads/{name}", param.grad.detach().cpu(), epoch)

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

                    loss = self.compute2dBatchLoss(batch_idx, batch_data, self.batch_size,config,epoch,validation_loop=True)
                    
                    # Log the validation loss
                    print("Validation - Epoch {} Batch {} Loss: {:.4f}".format(
                        epoch, batch_idx, loss.item()))
                    if self.val_writer is not None:
                        self.val_writer.add_scalar('loss', loss.item(), epoch * len(self.val_loader) + batch_idx)
                        self.val_writer.flush()
                    val_loss += loss.item()

            # Calculate average validation loss    
            avg_val_loss = val_loss / len(self.val_loader)
            print(f"Epoch {epoch} Average Validation Loss: {avg_val_loss:.4f}")
            self.log_epoch_metrics(epoch, avg_train_loss=avg_training_loss , avg_val_loss=avg_val_loss)

            # --- CHECKPOINTING ---
            # always save "last", each epoch
            self.save_checkpoint(last_path,epoch,global_step,avg_val_loss,config)

            # save best model if improved
            if best_val is None or avg_val_loss < best_val:
                best_val = avg_val_loss
                self.save_checkpoint(best_path,epoch,global_step,avg_val_loss,config)
                if self.val_writer is not None:
                    self.val_writer.add_scalar("Checkpoint/best_val_loss", best_val, epoch)
            # save periodic checkpoints every K_PERIOD epochs
            if (epoch + 1) % K_PERIOD == 0:
                periodic_path = periodic_tmpl.format(epoch=epoch + 1)
                self.save_checkpoint(periodic_path,epoch,global_step,avg_val_loss,config)
            # cleanup old periodic checkpoints, keeping only the newest KEEP_PERIOD
            self.cleanup_old(periodic_tmpl + '*', keep=KEEP_PERIODIC)

            # log the "best so far" marker, related to early stopping
            if self.val_writer is not None and early_stopper.best is not None:
                self.val_writer.add_scalar("EarlyStopping/best_val_loss", early_stopper.best, epoch)
                self.val_writer.add_scalar("EarlyStopping/patience_used", early_stopper.num_bad_epochs, epoch)

            if ray_tune:
                train.report({
                    "training_loss":float(avg_training_loss),
                    "validation_loss":float(avg_val_loss), 
                    "epoch":int(epoch)
                })
            
            # ---- EARLY STOPPING HOOK ----
            stop_now = early_stopper.step(avg_val_loss, self.model, self.optimizer, epoch)
            if stop_now:
                print(f"[EarlyStopping] No improvement for {early_stopper.patience} epochs. "
                    f"Best epoch was {early_stopper.best_epoch} with val={early_stopper.best:.6f}. "
                    f"Stopping at epoch {epoch}.")
                break
        # ===== After the for-epoch loop finishes =====
        ckpt = early_stopper.restore(self.model, self.optimizer)
        if ckpt is not None:
            print(f"[EarlyStopping] Restored best model from epoch {ckpt['epoch']} "
                f"({ckpt['monitored_value']:.6f}).")
        # Optional: log the chosen/best epoch to TB
        if self.val_writer is not None:
            self.val_writer.add_scalar("EarlyStopping/best_epoch", ckpt["epoch"], ckpt["epoch"])       

    def main(self):
        log.info("Starting SMALabellerApp with unet_dimensions: {}".format(self.unet_dimensions))
        if self.unet_dimensions == '2D':
            
            config = {
                "lr": 0.000023173,
                "max_lr": 0.0051206,
                "loss_function": 'cross_entropy',
                "augmentations": True,
                "dropout_rate": 0.45012
            }
            self.do2dTraining(config)
            
        """
        elif self.unet_dimensions == '3D':
            self.model = self.init3dModel()
            self.optimizer, self.scheduler = self.initOptimizer()
            self.initTensorboardWriters()
            self.do3dTraining()
        """
SMALabellerApp().main()












