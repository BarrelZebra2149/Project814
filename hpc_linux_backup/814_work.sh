#!/bin/bash
#PBS -N work814
#PBS -q iworkq
#PBS -l select=1:ncpus=32:mpiprocs=32:ngpus=2:mem=360gb
#PBS -l walltime=24:00:00

source ${HOME}/JupyterLAB/bin/activate
# pip install -r requirements.txt
# pip install --upgrade pip setuptools wheel
python 814_solution_double_fitness.py
