#!/bin/bash
### Job Parameters:
# basic info
#SBATCH --job-name "frechet"               # name
#SBATCH --output "Frechet.out"      # output file
#SBATCH --error "Frechet.err"       # error message file

# resource request info 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user bewagner@davidson.edu

## Script to Execute:
# change working directory to pipenv managed directory

cd ~/TPC-SnowflakeNet/completion
source /opt/conda/bin/activate pointnet

# execute python script in virtal env.
python3 frechet.py