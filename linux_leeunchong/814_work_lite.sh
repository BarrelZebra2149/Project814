#!/bin/bash
#PBS -N work814
#PBS -q workq
#PBS -l select=1:ncpus=8:mpiprocs=8:ngpus=1:mem=8gb
#PBS -l walltime=48:00:00

source ${HOME}/JupyterLAB/bin/activate
# pip install -r requirements.txt
# pip install --upgrade pip setuptools wheel
python 814_solution_gpu_full.py
