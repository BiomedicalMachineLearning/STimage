#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=150000
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --job-name ViT_NB
#SBATCH --array=0-11
module load cuda/11.0.2.450
module load gnu7
module load openmpi3
module load anaconda/3.6
source activate /scratch/imb/Xiao/.conda/envs/stimage

PATH=/scratch/imb/Xiao/.conda/envs/stimage/bin:$PATH

cd /clusterdata/uqxtan9/Xiao/STimage/development

python ./stimage_compare_histogene_1000hvg/ViT_NB_1000hvg.py $SLURM_ARRAY_TASK_ID
