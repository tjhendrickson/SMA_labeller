#!/bin/bash

########################################################################################################################
# To modify SMA labels used for SMA labeller model
# Author - Timothy Hendrickson

########################################################################################################################

module load s5cmd
module load fsl
output_dir="/home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/modified_labels/316_CBIT/derivatives/abcd-hcp-pipeline"
nnUNet_image_dir="/home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_raw_data/Dataset503_MICCAI2012_and_316_CBIT_SMA_GM_ribbon"

for subj_id in `s5cmd ls s3://hendr522-316-cbit/derivatives/abcd-hcp-pipeline/sub | awk '{print $2}' | awk -F"/" '{print $1}'`; do
    shortened_subj_id=`echo ${subj_id} | awk -F"-" '{print $2}'`
    ses_ids=`s5cmd ls s3://hendr522-316-cbit/derivatives/abcd-hcp-pipeline/${subj_id}/ses | awk '{print $2}' | awk -F"/" '{print $1}'`
    first_ses_id=`echo ${ses_ids} | awk '{print $1}'`
    path_to_image=`ls ${nnUNet_image_dir}/images*/${shortened_subj_id}_0000.nii.gz`
    path_to_label=`ls ${nnUNet_image_dir}/labels*/${shortened_subj_id}.nii.gz`
    if [ ! ${path_to_image} = "" ]; then
        ${subj_id} ${first_ses_id}
        mkdir -p ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/
        s5cmd cp s3://hendr522-316-cbit/derivatives/abcd-hcp-pipeline/${subj_id}/${first_ses_id}/files/MNINonLinear/ribbon.nii.gz ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/
        s5cmd cp s3://hendr522-316-cbit/derivatives/abcd-hcp-pipeline/${subj_id}/${first_ses_id}/files/MNINonLinear/T1w_restore.nii.gz ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/
        fslmaths ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/ribbon.nii.gz -uthr 3 -thr 3 -bin ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/lh_GM_ribbon_binary_mask.nii.gz
        fslmaths ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/ribbon.nii.gz -uthr 42 -thr 42 -bin ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/rh_GM_ribbon_binary_mask.nii.gz
        fslmaths ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/lh_GM_ribbon_binary_mask.nii.gz -add ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/rh_GM_ribbon_binary_mask.nii.gz ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/lh+rh_GM_ribbon_binary_mask.nii.gz
        fslmaths ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/lh+rh_GM_ribbon_binary_mask.nii.gz -mul /home/cconelea/shared/projects/316_CBIT/BIDS_output/derivatives/TMS_Targeting/harvardoxford-cortical_ROI-SMA_tpl-1mmMNI152.nii.gz ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/lh+rh_GM_ribbon_SMA_binary_mask.nii.gz
        cp ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/T1w_restore.nii.gz ${path_to_image}
        cp ${output_dir}/${subj_id}/${first_ses_id}/files/MNINonLinear/lh+rh_GM_ribbon_SMA_binary_mask.nii.gz ${path_to_label}
    fi
      
done
