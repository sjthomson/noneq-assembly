#!/bin/env bash
#SBATCH --job-name=param_sweep
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00
#SBATCH --account=semt035344
#SBATCH --array=1-4096%4096

module add languages/python/3.12.3

export WORK_DIR=/user/home/wl21287/clusterCode
cd ${WORK_DIR}

D_TAU_PAIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" coeffs.txt)

D=$(echo $D_TAU_PAIR | awk '{print $1}')
tau=$(echo $D_TAU_PAIR | awk '{print $2}')

echo JOB ID: ${SLURM_JOBID}
echo PBS ARRAY ID: ${SLURM_ARRAY_TASK_ID}

python3 main.py --D $D --tau $tau
