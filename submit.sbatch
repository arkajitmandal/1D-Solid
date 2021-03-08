#!/bin/bash
#SBATCH -p debug
#SBATCH -J solid 
#SBATCH -o my_output_%j
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 1:00:00
##SBATCH -N 1 
python3 solid.py