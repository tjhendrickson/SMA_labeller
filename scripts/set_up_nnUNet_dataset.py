#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:56:07 2024

Generate training and test set amenable for nnUNet training

@author: hendr522
"""

from sklearn.model_selection import train_test_split
from glob import glob
import os
import shutil

miccai_2012_dataset_dir='/panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/MICCAI2012_CompleteRelease/release/MICCAI-2012-Multi-Atlas-Challenge-Data'
nnUNet_training_dataset_dir='/home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnUNet_raw/Dataset001_MICCAI2012/'


training_label_paths = glob(os.path.join(miccai_2012_dataset_dir,'training-labels','*.nii.gz'))
testing_label_paths = glob(os.path.join(miccai_2012_dataset_dir,'testing-labels','*.nii.gz'))

full_dataset_labels_paths = training_label_paths + testing_label_paths


train_paths, test_paths = train_test_split(full_dataset_labels_paths,test_size=0.2)


for train_label_path in train_paths:
    subject_id=os.path.basename(train_label_path).split('_')[0]
    train_image_path=glob(os.path.join(miccai_2012_dataset_dir,'*images',subject_id+'_3.nii.gz'))[0]
    shutil.copy(train_image_path,os.path.join(nnUNet_training_dataset_dir,'imagesTr',subject_id+'_0000.nii.gz'))
    shutil.copy(train_label_path,os.path.join(nnUNet_training_dataset_dir,'labelsTr',subject_id+'.nii.gz'))
    os.system('fslmaths {label_path} -thr 192 -uthr 193 -bin {label_path}'.format(label_path=os.path.join(nnUNet_training_dataset_dir,'labelsTr',subject_id+'.nii.gz')))
    
for test_label_path in test_paths:
    subject_id=os.path.basename(test_label_path).split('_')[0]
    test_image_path=glob(os.path.join(miccai_2012_dataset_dir,'*images',subject_id+'_3.nii.gz'))[0]
    shutil.copy(test_image_path,os.path.join(nnUNet_training_dataset_dir,'imagesTs',subject_id+'_0000.nii.gz'))
    shutil.copy(test_label_path,os.path.join(nnUNet_training_dataset_dir,'labelsTs',subject_id+'.nii.gz'))
    os.system('fslmaths {label_path} -thr 192 -uthr 193 -bin {label_path}'.format(label_path=os.path.join(nnUNet_training_dataset_dir,'labelsTs',subject_id+'.nii.gz')))