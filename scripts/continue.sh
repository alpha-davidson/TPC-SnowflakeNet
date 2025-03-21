#!/bin/bash
### Job Parameters:
# basic info
#SBATCH --job-name "train-spd"               # name
#SBATCH --output "Continue-out.log"      # output file
#SBATCH --error "Continue-err.log"       # error message file

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
python train.py --config ./configs/22Mg.yaml --start_checkpoint ./exp/checkpoints/MyMgSampling-v3.0/ckpt-last.pth --resume --exp_name MyMgSampling-v3.0