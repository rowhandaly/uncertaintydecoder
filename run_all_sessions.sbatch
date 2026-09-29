#!/bin/bash
# Batch job for Rockfish: runs every session and window, then pools.
# Submit from the repository folder with:   sbatch run_all_sessions.sbatch
# Follow progress with:                     tail -f logs/run_all.txt
#
# The partition name and time limit may need adjusting for your account:
# `sinfo -s` lists the partitions you can use.
#SBATCH --job-name=uncertainty
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/run_all.txt

mkdir -p logs
python run_all_sessions.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary
