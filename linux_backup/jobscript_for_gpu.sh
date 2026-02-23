#!/bin/bash
#PBS -N jupyter814
#PBS -q iworkq
#PBS -l select=1:ncpus=64:mpiprocs=8:ngpus=1:mem=360gb
#PBS -l walltime=24:00:0
#############################################################
#environment variable set
USER_NAME="$USER"
CLUSTER_NAME=$(hostname -I | awk '{print $1}')
#############################################################
#venv activation
source ${HOME}/JupyterLAB/bin/activate
#############################################################
#set random port
PORT=$(shuf -i 8000-9999 -n 1)
NODE=$(hostname)
#############################################################
#create info file
INFOFILE="${HOME}/JupyterLAB/jupyter_info_${PBS_JOBID}.txt"
#############################################################
echo "Node: $NODE"    >> $INFOFILE
echo "Port: $PORT"    >> $INFOFILE
echo "To connect:"    >> $INFOFILE
echo "=== terminal : ssh tunneling  ===" >> $INFOFILE
echo "ssh -N -L ${PORT}:${NODE}:${PORT} ${USER_NAME}@${CLUSTER_NAME}" >> $INFOFILE
echo "=== local web browser information ===" >> $INFOFILE
echo "localhost:$PORT" >> $INFOFILE
# Jupyter Lab
nohup timeout 12h jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT} > jupyter.log 2>&1 &
