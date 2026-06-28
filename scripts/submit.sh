#!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=16
    #SBATCH --time=168:0:0
    source /opt/intel/oneapi/setvars.sh
    export OMP_NUM_THREADS=16
    cd "$SLURM_SUBMIT_DIR"
    python -u "$1"
