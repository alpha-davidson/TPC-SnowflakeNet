#!/bin/bash
### Job Parameters:
# basic info
#SBATCH --job-name "train-spd"                # name
#SBATCH --output "Train_JustMg512In2048OutEMD.out"      # output file
#SBATCH --error  "Train_JustMg512In2048OutEMD.err"      # error message file

# resource request info 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user bewagner@davidson.edu

## Script to Execute:
# change working directory to pipenv managed directory

cd completion
source /opt/conda/bin/activate spd
# module use /opt/pub/eb/modules/casecadelake/Core
# module load fosscuda #OpenBLAS

# execute python script in virtal env.
python train.py --config ./configs/JustMgCDL1.yaml --exp_name JustMg512In2048OutEMD
