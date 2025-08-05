import os
from glob import glob
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm3d,BatchNorm2d
from torch.nn.functional import leaky_relu
from torch.utils.data import Dataset

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityd,
    RandFlipd, RandAffined, RandGaussianNoised, RandCropByPosNegLabeld, CastToTyped)
from monai.data import CacheDataset
from monai.losses import DiceLoss, DiceCELoss

import diskcache as dc

import pdb

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
home_dir = os.path.expanduser('~')
#loss_fn = DiceLoss(sigmoid=True,reduction="mean",include_background=True)
loss_fn = DiceCELoss(sigmoid=True, reduction="mean", include_background=True)

def center_crop_3d(in_tensor, target_size):
    _, _, h, w, d = in_tensor.size()
    _, _, th, tw, td = target_size.size()
    x1 = (h - th) // 2
    y1 = (w - tw) // 2
    z1 = (d - td) // 2
    return in_tensor[:, :, x1:x1+th, y1:y1+tw, z1:z1+td]

def center_crop_2d(tensor, target_tensor):
    _, _, h, w = tensor.size()
    _, _, th, tw = target_tensor.size()
    x1 = (h - th) // 2
    y1 = (w - tw) // 2
    return tensor[:, :, x1:x1+th, y1:y1+tw]

def pad_to_match_3d(in_tensor, target_size):
    _, _, h, w, d = in_tensor.size()
    _, _, th, tw, td = target_size.size()
    diffX = th - h
    diffY = tw - w
    diffZ = td - d
    return F.pad(in_tensor, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2,
                          diffZ // 2, diffZ - diffZ // 2])

def pad_to_match_2d(output, reference):
    diffY = reference.size()[2] - output.size()[2]
    diffX = reference.size()[3] - output.size()[3]
    return F.pad(output, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def MONAIDiceLoss(inputs, targets):
    loss = loss_fn(inputs, targets)
    return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')  # expects raw logits

    def forward(self, inputs, targets):

        # compute binary cross-entropy loss from logits
        bce_loss = self.bce(inputs, targets.float())
        
        # Apply sigmoid since inputs are logits
        inputs = torch.sigmoid(inputs)       

        # Flatten
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        dice_loss = 1 - dice  # Dice loss is 1 - Dice coefficient

        return dice_loss + bce_loss  # Combine BCE and Dice loss

class Custom3DMRIDatasetMONAI:
    def __init__(
        self,
        data_dicts,
        augmentations=True,
        cache_rate=1.0,
        num_workers=4,
        pixdim=(1.0, 1.0, 1.0)):
        self.data_dicts = data_dicts
        self.augmentations = augmentations
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.pixdim = pixdim

        # Compose transforms
        self.transforms = self._build_transforms()

        # Create MONAI CacheDataset
        self.dataset = CacheDataset(
            data=self.data_dicts,
            transform=self.transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers)

    def _build_transforms(self):
        base_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=self.pixdim, mode=["bilinear", "nearest"]),
            ScaleIntensityd(keys="image"),
            CastToTyped(keys=["label"], dtype=np.float32)
        ]
        if self.augmentations:
            base_transforms += self._add_augmentations()
        return Compose(base_transforms)
    
    def _add_augmentations(self):
        augmentations = [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandAffined(keys=["image", "label"], mode=["bilinear","nearest"], prob=0.5, rotate_range=(np.pi/18, np.pi/18, np.pi/18), scale_range=(0.1, 0.1, 0.1)),
            RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=0.1),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image"
            )]

        return augmentations

    def get_dataset(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class Custom3DMRIImageDataset(Dataset):
    nib.Nifti1Header.quaternion_threshold = -1e-06
    def __init__(self,img_dir,annotations_dir,augmentations=None,cache_dir=os.path.join(home_dir,'.mri_cache')):
        self.annotations_dir = annotations_dir
        self.img_dir = img_dir
        self.augmentations = augmentations
        self.cache = dc.Cache(cache_dir)

    def __len__(self):
        files = [f for f in os.listdir(self.annotations_dir)  if os.path.isfile(os.path.join(self.annotations_dir, f)) and '.nii' in f ]
        return len(files)

    def __getitem__(self,idx):
        # image and label paths
        label_images = sorted(glob(os.path.join(self.annotations_dir,'*.nii.gz')))
        label_image_path = label_images[idx]
        image_name = os.path.basename(label_image_path).split('.')[0]+'_0000.nii.gz'
        image_path = os.path.join(self.img_dir,image_name)

        # create cache keys
        label_cache_key = f"mri_label_{os.path.basename(label_image_path)}"
        image_cache_key = f"mri_image_{os.path.basename(image_path)}"

        # try to load label from cache
        if label_cache_key in self.cache:
            label_data_tensor = self.cache[label_cache_key]
        else:
            # load label image
            label_loaded = nib.load(label_image_path)
            label_data_np = label_loaded.get_fdata(dtype='single')

            # add channel dimension for 3d conv
            label_data_np = np.expand_dims(label_data_np, axis=0)

            # convert to pytorch tensor
            label_data_tensor = torch.from_numpy(label_data_np)

            # cache result
            self.cache[label_cache_key] = label_data_tensor

        # try to load image from cache
        if image_cache_key in self.cache:
            image_data_tensor = self.cache[image_cache_key]
        else:
            # load image
            image_loaded = nib.load(image_path)
            image_data_np = image_loaded.get_fdata(dtype='single')

            # add channel dimension for 3d conv
            image_data_np = np.expand_dims(image_data_np, axis=0)

            # convert to pytorch tensor
            image_data_tensor = torch.from_numpy(image_data_np)

            # cache result
            self.cache[image_cache_key] = image_data_tensor

        #if self.augmentations:

        return image_data_tensor, label_data_tensor


class UNet2D(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        forward_encode_1 = self.enc1(x)
        forward_encode_2 = self.enc2(self.pool1(forward_encode_1))
        forward_encode_3 = self.enc3(self.pool2(forward_encode_2))
        forward_encode_4 = self.enc4(self.pool3(forward_encode_3))
        forward_encode_5 = self.bottleneck(self.pool4(forward_encode_4))

        # Decoder
        forward_upconv_1 = self.upconv1(forward_encode_5)
        forward_encode_4_cropped = center_crop_2d(forward_encode_4, forward_upconv_1)
        forward_decode_1 = self.dec1(torch.cat([forward_upconv_1, forward_encode_4_cropped], dim=1))

        forward_upconv_2 = self.upconv2(forward_decode_1)
        forward_encode_3_cropped = center_crop_2d(forward_encode_3, forward_upconv_2)
        forward_decode_2 = self.dec2(torch.cat([forward_upconv_2, forward_encode_3_cropped], dim=1))

        forward_upconv_3 = self.upconv3(forward_decode_2)
        forward_encode_2_cropped = center_crop_2d(forward_encode_2, forward_upconv_3)
        forward_decode_3 = self.dec3(torch.cat([forward_upconv_3, forward_encode_2_cropped], dim=1))

        forward_upconv_4 = self.upconv4(forward_decode_3)
        forward_encode_1_cropped = center_crop_2d(forward_encode_1, forward_upconv_4)
        forward_decode_4 = self.dec4(torch.cat([forward_upconv_4, forward_encode_1_cropped], dim=1))

        out = self.outconv(forward_decode_4)
        return pad_to_match_2d(out, x)



class UNet3D(nn.Module):
    def __init__(self,n_class):
        super(UNet3D, self).__init__()

        #Encoder

        # each block of the encoder consists of two convolutional layers followed by a pooling layer

        #input: 182x218x182x1
        self.encode1layer1=nn.Conv3d(1,64,kernel_size=3,padding=1,stride=1) # output: 64,182,218,182
        self.encode1layer2=nn.Conv3d(64,64,kernel_size=3,padding=1,stride=1) # output: 64,182,218,182
        self.pool1 = nn.MaxPool3d(kernel_size=2,padding=1,stride=2) # output: 64,92,110,92

        self.encode2layer1=nn.Conv3d(64,128,kernel_size=3,padding=1,stride=1) # output: 128,92,110,92
        self.encode2layer2 = nn.Conv3d(128, 128, kernel_size=3, padding=1,stride=1,) # output: 128,92,110,92
        self.pool2 = nn.MaxPool3d(kernel_size=2,padding=1,stride=2) # output: 128,47,56,47

        self.encode3layer1=nn.Conv3d(128,256,kernel_size=3,padding=1,stride=1) # output: 256,47,56,47
        self.encode3layer2=nn.Conv3d(256,256,kernel_size=3,padding=1,stride=1) # output: 256,47,56,47
        self.pool3 = nn.MaxPool3d(kernel_size=2,stride=2,padding=1) # output: 256, 24, 29, 24

        self.encode4layer1=nn.Conv3d(256,512,kernel_size=3,padding=1,stride=1) # output: 512, 24, 29, 24
        self.encode4layer2=nn.Conv3d(512,512,kernel_size=3,padding=1,stride=1) # output: 512, 24, 29, 24

        # Decoder
        self.upconv1=nn.ConvTranspose3d(512,256,kernel_size=2,stride=2,padding=1) # output:  256, 46, 56, 46
        self.decode1layer1=nn.Conv3d(512,256,kernel_size=3,stride=1,padding=1) # output:  256, 46, 56, 46
        self.decode1layer2=nn.Conv3d(256,256,kernel_size=3,stride=1,padding=1) # output:  256, 46, 56, 46

        self.upconv2=nn.ConvTranspose3d(256,128,kernel_size=2,stride=2,padding=1) # output: 128, 90, 110, 90
        self.decode2layer1=nn.Conv3d(256,128,kernel_size=3,stride=1,padding=1) # output: 128, 90, 110, 90
        self.decode2layer2=nn.Conv3d(128,128,kernel_size=3,stride=1,padding=1) # output: 128, 90, 110, 90

        self.upconv3=nn.ConvTranspose3d(128,64,kernel_size=2,stride=2,padding=1) # output: 178, 218, 178
        self.decode3layer1=nn.Conv3d(128,64,kernel_size=3,stride=1,padding=1) # output: 64, 178, 218, 178
        self.decode3layer2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1) # output:  64, 178, 218, 178

        self.outconv = nn.Conv3d(64,n_class,kernel_size=1) # output: 1, 178, 218, 178

    def forward(self,x):

        # Encoder
        BN_encode1layer1 = BatchNorm3d(self.encode1layer1.out_channels).to(device)
        forward_encode1layer1= leaky_relu(BN_encode1layer1(self.encode1layer1(x)))
        BN_encode1layer2 = BatchNorm3d(self.encode1layer2.out_channels).to(device)
        forward_encode1layer2 = leaky_relu(BN_encode1layer2(self.encode1layer2(forward_encode1layer1)))
        forward_pool1 = self.pool1(forward_encode1layer2)

        BN_encode2layer1 = BatchNorm3d(self.encode2layer1.out_channels).to(device)
        forward_encode2layer1 = leaky_relu(BN_encode2layer1(self.encode2layer1(forward_pool1)))
        BN_encode2layer2 = BatchNorm3d(self.encode2layer2.out_channels).to(device)
        forward_encode2layer2 = leaky_relu(BN_encode2layer2(self.encode2layer2(forward_encode2layer1)))
        forward_pool2=self.pool2(forward_encode2layer2)

        BN_encode3layer1 = BatchNorm3d(self.encode3layer1.out_channels).to(device)
        forward_encode3layer1 = leaky_relu(BN_encode3layer1(self.encode3layer1(forward_pool2)))
        BN_encode3layer2 = BatchNorm3d(self.encode3layer2.out_channels).to(device)
        forward_encode3layer2 = leaky_relu(BN_encode3layer2(self.encode3layer2(forward_encode3layer1)))
        forward_pool3=self.pool3(forward_encode3layer2)

        BN_encode4layer1 = BatchNorm3d(self.encode4layer1.out_channels).to(device)
        forward_encode4layer1 = leaky_relu(BN_encode4layer1(self.encode4layer1(forward_pool3)))
        BN_encode4layer2 = BatchNorm3d(self.encode4layer2.out_channels).to(device)
        forward_encode4layer2 = leaky_relu(BN_encode4layer2(self.encode4layer2(forward_encode4layer1)))

        # Decoder
        forward_upconv1=self.upconv1(forward_encode4layer2)
        forward_encode3layer2_cropped = center_crop_3d(forward_encode3layer2, forward_upconv1)
        skipconnection1=torch.cat([forward_upconv1,forward_encode3layer2_cropped],dim=1)
        BN_decode1layer1 = BatchNorm3d(self.decode1layer1.out_channels).to(device)
        forward_decode1layer1=leaky_relu(BN_decode1layer1(self.decode1layer1(skipconnection1)))
        BN_decode1layer2=BatchNorm3d(self.decode1layer2.out_channels).to(device)
        forward_decode1layer2 = leaky_relu(BN_decode1layer2(self.decode1layer2(forward_decode1layer1)))

        forward_upconv2=self.upconv2(forward_decode1layer2)
        forward_encode2layer2_cropped = center_crop_3d(forward_encode2layer2, forward_upconv2)
        skipconnection2=torch.cat([forward_upconv2,forward_encode2layer2_cropped],dim=1)
        BN_decode2layer1 = BatchNorm3d(self.decode2layer1.out_channels).to(device)
        forward_decode2layer1=leaky_relu(BN_decode2layer1(self.decode2layer1(skipconnection2)))
        BN_decode2layer2 = BatchNorm3d(self.decode2layer2.out_channels).to(device)
        forward_decode2layer2 = leaky_relu(BN_decode2layer2(self.decode2layer2(forward_decode2layer1)))

        forward_upconv3=self.upconv3(forward_decode2layer2)
        forward_encode1layer2_cropped = center_crop_3d(forward_encode1layer2, forward_upconv3)
        skipconnection3=torch.cat([forward_upconv3,forward_encode1layer2_cropped],dim=1)
        BN_decode3layer1 = BatchNorm3d(self.decode3layer1.out_channels).to(device)
        forward_decode3layer1=leaky_relu(BN_decode3layer1(self.decode3layer1(skipconnection3)))
        BN_decode3layer2 = BatchNorm3d(self.decode3layer2.out_channels).to(device)
        forward_decode3layer2 = leaky_relu(BN_decode3layer2(self.decode3layer2(forward_decode3layer1)))

        # output layer

        out = self.outconv(forward_decode3layer2)
        out_cropped = pad_to_match_3d(out,x)
        return out_cropped