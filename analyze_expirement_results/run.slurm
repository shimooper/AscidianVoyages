#!/bin/bash

#SBATCH --job-name=ships
#SBATCH --account=power-general-users
#SBATCH --partition=power-general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --output=/groups/pupko/yairshimony/ships/%j.out
#SBATCH --error=/groups/pupko/yairshimony/ships/%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"
echo "Job name: $SLURM_JOB_NAME"

export HOME=/groups/pupko/yairshimony
source ~/miniconda3/etc/profile.d/conda.sh
conda activate microbializer
export PATH=$CONDA_PREFIX/bin:$PATH

python ~/ships/analyze_expirement_results/main.py --cpus $SLURM_JOB_CPUS_PER_NODE