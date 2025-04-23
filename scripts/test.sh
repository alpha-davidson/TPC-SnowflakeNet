#!/bin/bash
### Job Parameters:
# basic info
#SBATCH --job-name "test-spd"               # name
#SBATCH --output "TestMgAndOCDL1.out"      # output file
#SBATCH --error "TestMgAndOCDL1.err"       # error message file

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
# python3 test.py --config ./configs/MgAndOEMD.yaml --model ./exp/checkpoints/MgAndO512In2048Out/ckpt-best.pth
python3 test.py --config ./configs/MgAndOCDL1.yaml --model ./exp/checkpoints/MgAndO512In2048OutCDL1/ckpt-best.pth
# python3 test.py --config ./configs/JustMgEMD.yaml --model ./exp/checkpoints/JustMg512In2048OutEMD/ckpt-best.pth
# python3 test.py --config ./configs/JustMgCDL1.yaml --model ./exp/checkpoints/JustMg512In2048OutCDL1/ckpt-best.pth