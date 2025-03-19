#!/bin/bash
### Job Parameters:
# basic info
#SBATCH --job-name "emd_testing"               # name
#SBATCH --output "EMD4dtest-out.log"      # output file
#SBATCH --error "EMD4dtest-err.log"       # error message file

# resource request info 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user bewagner@davidson.edu

## Script to Execute:
# change working directory to pipenv managed directory

cd ~/TPC-SnowflakeNet
source /opt/conda/bin/activate spd
# module use /opt/pub/eb/modules/casecadelake/Core
# module load fosscuda #OpenBLAS

# execute python script in virtal env.
cd models/pointnet2_ops_lib
python setup.py install

cd ../..

cd loss_functions/Chamfer3D
python setup.py install

cd ../Chamfer4D
python setup.py install

cd ../emd4D
python setup.py install
# python 4demd_module.py

cd ../emd
python setup.py install