#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/output_terminal/%j.out
#SBATCH --job-name=clipseg_phrasecut
#SBATCH --gres=gpu
#SBATCH --mem=10000
#SBATCH --mail-user=oluwatosin.alabi@kcl.ac.uk

source /users/${USER}/.bashrc
source activate MTPSL
python training.py phrasecut.yaml 0