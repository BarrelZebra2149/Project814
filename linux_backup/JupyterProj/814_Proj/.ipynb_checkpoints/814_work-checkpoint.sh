#!/bin/bash
#PBS -N jupyter814
#PBS -q iworkq
#PBS -l select=1:ncpus=32:mpiprocs=32:ngpus=2:mem=360gb
#PBS -l walltime=24:00:00

pip install -r requirements.txt
pip install --upgrade pip setuptools wheel
python -m scoop ga_scoop.py > output.txt
