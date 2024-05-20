#!/bin/bash

#SBATCH --job-name="DUST3R"

#SBATCH --output=/srv/essa-lab/flash3/vcartillier3/egoexo-view-synthesis/dependencies/dust3r/batch_scripts/slurm_logs/sample-%j.out

#SBATCH --error=/srv/essa-lab/flash3/vcartillier3/egoexo-view-synthesis/dependencies/dust3r/batch_scripts/slurm_logs/sample-%j.err

## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=1

#SBATCH --gpus-per-node=a40:2

#SBATCH --cpus-per-task=12

#SBATCH -p essa-lab

#SBATCH --qos short

srun $1 $2
