#!/bin/bash
#PBS -N work814
#PBS -q iworkq
#PBS -l select=1:ncpus=64:mpiprocs=64:ngpus=1:mem=16gb
#PBS -l walltime=48:00:00

source ${HOME}/JupyterLAB/bin/activate
# pip install -r requirements.txt
# pip install --upgrade pip setuptools wheel
python 814_solution_score_first.py
python 814_solution_count_first.py
