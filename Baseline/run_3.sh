#!/bin/bash
#
#SBATCH --job-name=Baseline
#SBATCH --output=/mnt/beegfs/work/baseline/res_2.txt
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu_model:a180"


source /storage-1/miniconda3/bin/activate


srun python /mnt/beegfs/work/baseline/scrum_baseline_vllm_faster_2.py
