#!/bin/bash

miccai_2012_dataset_dir="/panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/MICCAI2012_CompleteRelease/release/MICCAI-2012-Multi-Atlas-Challenge-Data"

for train_image in `ls ${miccai_2012_dataset_dir}/training-images/*.nii.gz`; do
	subj_id=`basename ${train_image} | awk -F"_" '{print $1}'`
	train_label=`ls ${miccai_2012_dataset_dir}/training-labels/${subj_id}_3_glm.nii.gz`