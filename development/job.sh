#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 28
#SBATCH --mem=300000
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:4
#SBATCH --job-name CNN_NB

cd /clusterdata/uqxtan9/Xiao/STimage/development

module load cuda/11.0.2.450
module load gnu7
module load openmpi3
module load anaconda/3.6
source activate /scratch/imb/Xiao/.conda/envs/stimage

python CNN_NB_train.py
