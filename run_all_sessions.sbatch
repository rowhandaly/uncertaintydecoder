#!/bin/bash -l
# Batch job for Rockfish: runs every session and window, then pools.
# Submit FROM THE REPOSITORY FOLDER (the job runs where it was submitted):
#     cd ~/uncertaintydecoder && sbatch run_all_sessions.sbatch
# Check it's queued/running:   squeue -u $USER
# Follow progress:             tail -f logs/run_all.txt
#
# "shared" is the partition for small jobs: 2 cores on a node shared with
# other people's jobs, rather than a whole node. `sinfo -s` lists the
# partitions. If the job runs out of time, just submit it again: finished
# steps are skipped, so it carries on where it stopped.
# If sbatch asks for an account, add a line:
#     #SBATCH --account=<your allocation name>
#SBATCH --job-name=uncertainty
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_all.txt

# The login shell loads Rockfish's default modules, including an older
# gcc whose C++ library (libstdc++) shadows anaconda's newer one and
# breaks pandas ("GLIBCXX_3.4.29 not found"). Unload the modules and put
# anaconda's own libraries first.
module purge 2>/dev/null
export LD_LIBRARY_PATH="$HOME/anaconda3/lib:$LD_LIBRARY_PATH"

echo "started $(date) on $(hostname)"
echo "python: $(which python)"
python -c "import torch, pandas, sklearn; print('packages ok')"

mkdir -p logs
python run_all_sessions.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary
echo "finished $(date)"
