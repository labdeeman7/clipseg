#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/output_terminal/clipseg_%j.out
#SBATCH --job-name=clipseg_phrasecut
#SBATCH --gres=gpu
#SBATCH --mem=10000
#SBATCH --mail-user=oluwatosin.alabi@kcl.ac.uk

source /users/${USER}/.bashrc
source activate CLIPSEG
python score.py endovis2017.yaml 0 0