#!/bin/bash
#SBATCH --account=wetosa  # allocation account
#SBATCH --time=02:00:00  # walltime
#SBATCH --job-name=transmission_064  # job name
#SBATCH --nodes=1  # number of nodes
#SBATCH --output=./stdout/transmission_064_%j.o
#SBATCH --error=./stdout/transmission_064_%j.e
#SBATCH --mem=90000  # node RAM in MB
echo Running on: $HOSTNAME, Machine Type: $MACHTYPE
echo source activate revruns
source activate revruns
echo conda env activate complete!

rrconnections -r 64 -a wetosa -t 2 -m 179