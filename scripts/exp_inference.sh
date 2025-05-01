#!/bin/bash
### Job Parameters:
# basic info
#SBATCH --job-name "exp-inference-spd"               # name
#SBATCH --output "ExpInference.out"      # output file
#SBATCH --error "ExpInference.err"       # error message file

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
python exp_inference.py --config ./configs/Mg22.yaml --model ./exp/checkpoints/JustMg512In2048OutCDL1/ckpt-best.pth --n_imgs "100" --save_img_path ../imgs/Mg22_exp/JustMg512In2048OutCDL1/
# python exp_inference.py --config ./configs/Mg22.yaml --model ./exp/checkpoints/JustMg512In2048OutEMD/ckpt-best.pth --n_imgs "100" --save_img_path ../imgs/Mg22_exp/JustMg512In2048OutEMD/
# python exp_inference.py --config ./configs/Mg22.yaml --model ./exp/checkpoints/MgAndO512In2048OutCDL1/ckpt-best.pth --n_imgs "100" --save_img_path ../imgs/Mg22_exp/MgAndO512In2048OutCDL1/
# python exp_inference.py --config ./configs/Mg22.yaml --model ./exp/checkpoints/MgAndO512In2048Out/ckpt-best.pth --n_imgs "100" --save_img_path ../imgs/Mg22_exp/MgAndO512In2048OutEMD/