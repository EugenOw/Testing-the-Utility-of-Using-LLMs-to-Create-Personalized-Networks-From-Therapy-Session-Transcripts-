#!/bin/bash
#SBATCH --job-name=pbt_experiment
#SBATCH --output=/mnt/beegfs/work/pbt/res.txt
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu_model:a180"

source /storage-1/miniconda3/bin/activate /storage-1/miniconda3/envs/my_first_env

srun python /mnt/beegfs/work/pbt/WP0.py
