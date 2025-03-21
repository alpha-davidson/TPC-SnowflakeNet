#!/bin/bash
### Job Parameters:
# basic info
#SBATCH --job-name "predict"               # name
#SBATCH --output "Predict.out"      # output file
#SBATCH --error "Predict.err"       # error message file

# resource request info 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user bewagner@davidson.edu

## Script to Execute:
# change working directory to pipenv managed directory

cd ~/TPC-SnowflakeNet/completion
source /opt/conda/bin/activate spd

# execute python script in virtal env.
python3 predict.py --config ./configs/MgAndO.yaml --model ./exp/checkpoints/MgOEmd/ckpt-best.pth --gt_save_path ./MgO_gts.npy --pred_save_path ./MgO.npy