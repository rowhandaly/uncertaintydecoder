#!/bin/bash -l
# Submit from the core folder, on a login node:  sbatch run_all.sbatch
#SBATCH --job-name=core
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/run_all.txt

# Rockfish's default modules shadow anaconda's C++ library and break pandas.
module purge 2>/dev/null
export LD_LIBRARY_PATH="$HOME/anaconda3/lib:$LD_LIBRARY_PATH"

mkdir -p logs
echo "started $(date)"
python run_all.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary
echo "finished $(date)"
