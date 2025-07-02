#!/bin/bash
sbatch <<EOT
#!/bin/sh

### Argument to this script is the fold number (between 0 and 4 
### inclusive) and -A argument 
### Sample invocation: ./NnUnetTrain_agate.sh 0 faird 3d_fullres 001 [-c]

#SBATCH --job-name=$4_$3_$1_Train_nnUNet # job name

#SBATCH --mem=40g        
#SBATCH --time=96:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p a100-4  
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=10               # total number of tasks across all nodes

#SBATCH --mail-type=begin        # send 7mail when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=hendr522@umn.edu
#SBATCH -e output_logs/Train_$1_$3_$4_SMA_labeller-%j.err
#SBATCH -o output_logs/Train_$1_$3_$4_SMA_labeller-%j.out

#SBATCH -A $2

## build script here
module load gcc cuda/12.0
source /users/3/hendr522/SW/miniconda3/etc/profile.d/conda.sh
conda activate nnUNet_v2


export nnUNet_raw="/home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_raw_data"
export nnUNet_preprocessed="/home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_preprocessed"
export nnUNet_results="/home/cconelea/shared/projects/316_CBIT/BIDS_output/code/SMA_labeller/data/nnUNet/nnunet_v2/nnUNet_results"

nnUNetv2_train $4 $3 $1 --npz -device cuda
EOT
