#!/bin/bash

module load fsl
dataset_path='/home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/pytorch_model/Dataset503_MICCAI2012_and_316_CBIT_SMA_GM_ribbon/'

for imageTr_path in `ls ${dataset_path}/imagesTr/????_0000.nii.gz`; do 
	filename=`basename ${imageTr_path} | awk -F'.' '{print $1}'`; 
	echo flirt -in ${imageTr_path} -ref /common/software/install/migrated/fsl/6.0.4/data/standard/MNI152_T1_1mm.nii -out ${dataset_path}/imagesTr_resampled/${filename} -omat ${dataset_path}/Tr_resample_xfms/${filename}.mat -dof 12;
  flirt -in ${imageTr_path} -ref /common/software/install/migrated/fsl/6.0.4/data/standard/MNI152_T1_1mm.nii -out ${dataset_path}/imagesTr_resampled/${filename} -omat ${dataset_path}/Tr_resample_xfms/${filename}.mat -dof 12;
	label_filename=`echo ${filename} | awk -F"_" '{print $1}'`; 
	label_file_path=`ls ${dataset_path}/labelsTr/${label_filename}.nii.gz`; 
	echo flirt -in ${label_file_path} -ref /common/software/install/migrated/fsl/6.0.4/data/standard/MNI152_T1_1mm.nii -out ${dataset_path}/labelsTr_resampled/${label_filename} -init ${dataset_path}/Tr_resample_xfms/${filename}.mat -applyxfm -interp nearestneighbour; 
  flirt -in ${label_file_path} -ref /common/software/install/migrated/fsl/6.0.4/data/standard/MNI152_T1_1mm.nii -out ${dataset_path}/labelsTr_resampled/${label_filename} -init ${dataset_path}/Tr_resample_xfms/${filename}.mat -applyxfm -interp nearestneighbour; 
done

for imageTs_path in `ls ${dataset_path}/imagesTs/????_0000.nii.gz`; do 
  filename=`basename ${imageTs_path} | awk -F'.' '{print $1}'`; 
  echo flirt -in ${imageTs_path} -ref /common/software/install/migrated/fsl/6.0.4/data/standard/MNI152_T1_1mm.nii -out ${dataset_path}/imagesTs_resampled/${filename} -omat ${dataset_path}/Ts_resample_xfms/${filename}.mat -dof 12;
  flirt -in ${imageTs_path} -ref /common/software/install/migrated/fsl/6.0.4/data/standard/MNI152_T1_1mm.nii -out ${dataset_path}/imagesTs_resampled/${filename} -omat ${dataset_path}/Ts_resample_xfms/${filename}.mat -dof 12;
  label_filename=`echo ${filename} | awk -F"_" '{print $1}'`; 
  label_file_path=`ls ${dataset_path}/labelsTs/${label_filename}.nii.gz`; 
  echo flirt -in ${label_file_path} -ref /common/software/install/migrated/fsl/6.0.4/data/standard/MNI152_T1_1mm.nii -out ${dataset_path}/labelsTs_resampled/${label_filename} -init ${dataset_path}/Ts_resample_xfms/${filename}.mat -applyxfm -interp nearestneighbour; 
  flirt -in ${label_file_path} -ref /common/software/install/migrated/fsl/6.0.4/data/standard/MNI152_T1_1mm.nii -out ${dataset_path}/labelsTs_resampled/${label_filename} -init ${dataset_path}/Ts_resample_xfms/${filename}.mat -applyxfm -interp nearestneighbour; 
done


