#!/bin/bash

#SBATCH --job-name=ships
#SBATCH --account=gpu-research
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --output=/home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_noa_figures/%j.out
#SBATCH --error=/home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_noa_figures/%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"
echo "Job name: $SLURM_JOB_NAME"

export CUDA_VISIBLE_DEVICES=""
export HOME=/home/ai_center/ai_users/yairshimony
source ~/miniconda/etc/profile.d/conda.sh
conda activate ships
export PATH=$CONDA_PREFIX/bin:$PATH

python ~/ships/analyze_expirement_results/main.py --cpus $SLURM_JOB_CPUS_PER_NODE --outputs_dir_name outputs_noa_figures --run_configurations_in_parallel False --run_lstm_configurations_in_parallel False