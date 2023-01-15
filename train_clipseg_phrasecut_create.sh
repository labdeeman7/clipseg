#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/output_terminal/clipseg_%j.out
#SBATCH --job-name=clipseg_phrasecut
#SBATCH --gres=gpu
#SBATCH --mail-user=oluwatosin.alabi@kcl.ac.uk

source /users/${USER}/.bashrc
source activate CLIPSEG
python training.py phrasecut.yaml 0