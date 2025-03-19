#!/bin/bash
### Job Parameters:
# basic info
#SBATCH --job-name "inference-spd"               # name
#SBATCH --output "Inference-out.log"      # output file
#SBATCH --error "Inference-err.log"       # error message file

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
python inference.py --config ./configs/VarInLen.yaml --model ./exp/checkpoints/JustMgv2.0/ckpt-best.pth --save_img_path /home/DAVIDSON/bewagner/TPC-SnowflakeNet/imgs/JustMgReal/ --n_imgs "100"