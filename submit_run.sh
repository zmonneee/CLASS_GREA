#!/bin/bash
#SBATCH --job-name=ccg
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=msq26sd@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

RUNDIR="$1"
YAML="$2"

PROJ=/mnt/parscratch/users/msq26sd/GREA

module load Python/3.10.8-GCCcore-12.2.0
module load mpi4py/3.1.4-gompi-2022b
module load OpenBLAS/0.3.21-GCC-12.2.0
module load CFITSIO/4.2.0-GCCcore-12.2.0
module load h5py/3.8.0-foss-2022b

source /users/msq26sd/GREA/venv/bin/activate
export COBAYA_PACKAGES_PATH=$PROJ/cobaya_packages
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$PROJ/$RUNDIR"
srun --export=ALL --cpus-per-task=$SLURM_CPUS_PER_TASK python -m cobaya run "$YAML" --resume

