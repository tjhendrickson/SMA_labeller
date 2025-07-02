#!/bin/bash
sbatch <<EOT
#!/bin/sh

### Argument to this script is the fold number (between 0 and 4 
### inclusive) and -A argument 
### Sample invocation: ./NnUnetPredict_agate.sh

#SBATCH --job-name=Predict_Dataset500_MICCAI2012_nnUNet # job name

#SBATCH --mem=40g        
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p a100-4  
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=10               # total number of tasks across all nodes

#SBATCH --mail-type=begin        # send 7mail when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=hendr522@umn.edu
#SBATCH -e output_logs/Predict_Dataset500_MICCAI2012_nnUNet_SMA_labeller-%j.err
#SBATCH -o output_logs/Predict_Dataset500_MICCAI2012_nnUNet_SMA_labeller-%j.out
#SBATCH -A cconelea

## build script here
module load gcc cuda/12.0
source /users/3/hendr522/SW/miniconda3/etc/profile.d/conda.sh
conda activate nnUNet_v2


export nnUNet_raw="/home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_raw_data"
export nnUNet_preprocessed="/home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_preprocessed"
export nnUNet_results="/home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_results"

nnUNetv2_predict -d Dataset500_MICCAI2012 -i /panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_raw_data/Dataset500_MICCAI2012/imagesTs \
-o /panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_results/Dataset500_MICCAI2012_predictions/predictions \
-f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans --save_probabilities --verbose -device cuda

nnUNetv2_apply_postprocessing -i /panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_results/Dataset500_MICCAI2012_predictions/predictions \
-o /panfs/jay/groups/6/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_results/Dataset500_MICCAI2012_predictions/predictions_postprocessing \
-pp_pkl_file /home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_results/Dataset500_MICCAI2012/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
-np 8 -plans_json /home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_results/Dataset500_MICCAI2012/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json
EOT
