#!/bin/bash
#SBATCH --job-name=MMtrain
#SBATCH --time=48:00:00
#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=200G

echo 'Running in:' $(pwd)
echo "On node: $(hostname)"
source ~/.bash_profile

# Exit immediately if any command has a non-zero return code.
set -e
set -o pipefail

CONDAENV=$1
CONFIG=$2
echo "Activate $CONDAENV"
conda activate $CONDAENV

CMD="bash tools/dist_train.sh $CONFIG $SLURM_GPUS ${@:3}"
echo "Launching command: $CMD"
eval "$CMD" || echo "Process exited with code: $?"
