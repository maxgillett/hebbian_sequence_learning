#!/bin/bash
#SBATCH --job-name=job_128
#SBATCH --output=logs/output_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 5760
#SBATCH --mem=16000
#SBATCH -e logs/err_%j.out
#module load Python/3.6.4
python3 figures/article/3/data_c.py -P 128 -i $i -c 20
