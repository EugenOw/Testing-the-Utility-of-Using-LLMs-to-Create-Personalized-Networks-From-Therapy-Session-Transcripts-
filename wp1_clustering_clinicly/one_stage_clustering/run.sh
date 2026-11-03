#!/bin/bash
#
#SBATCH --job-name=Clustering_experiment
#SBATCH --output=/mnt/beegfs/work/clustering/res.txt
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu_model:a180"


source /storage-1/miniconda3/bin/activate /storage-1/miniconda3/envs/thesis


srun python /mnt/beegfs/work/clustering/scrum_clustering.py
