#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16GB
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --job-name=4_rng
#SBATCH --partition=gpu_cuda
#SBATCH --account=a_nguyen_quan
#SBATCH --gres=gpu:h100:1 #you can ask for up to 2 here
#SBATCH --time=88:00:00
#SBATCH --array=8,9,10

# module load anaconda/3.6
module load anaconda3/2022.05
source activate /scratch/project_mnt/S0010/Jacky/envs/xenium
# module load cuda/10.0.130
# module load gnu/5.4.0
# module load mvapich2
# module load git-2.19.1-gcc-7.3.0-swjt5hp

export PATH=/scratch/project_mnt/S0010/Jacky/envs/xenium:$PATH
export PATH=/scratch/project_mnt/S0010/Jacky/envs/xenium/bin:$PATH

cd /scratch/project_mnt/S0010/Jacky/scripts

# Params are:
# Lr = 1e5 = 0.1 * 1e4 (end lr from base hovernet)
# Weight decay = 0 (from base hovernet)
# Step gamma = 0.5 (from 0.1 base hovernet)

python train_hovernet_rep1.py \
    --dataset /scratch/project_mnt/S0010/Jacky/tile_data/QMDL01,/scratch/project_mnt/S0010/Jacky/tile_data/QMDL02,/scratch/project_mnt/S0010/Jacky/tile_data/QMDL03,/scratch/project_mnt/S0010/Jacky/tile_data/QMDL04 \
    --out-dir /scratch/project_mnt/S0010/Xiao/hovernet_out_$SLURM_ARRAY_TASK_ID \
    --lr 0.0003 \
    --step-gamma 0.1 \
    --weight-decay 0.0 \
    --n-classes 5 \
    --split 0.85,0.15,0.00 \
    --batch-size 24 \
    --mod \
    --seed $SLURM_ARRAY_TASK_ID
