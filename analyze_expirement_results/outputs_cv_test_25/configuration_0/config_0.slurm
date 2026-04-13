#!/bin/bash 
#SBATCH --job-name=config_0
#SBATCH --account=gpu-research
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_cv3_mcc_fix_plots_and_scaler_and_stratify/logs/config_0_%j.out
#SBATCH --error=/home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_cv3_mcc_fix_plots_and_scaler_and_stratify/logs/config_0_%j.err
echo Job ID: $SLURM_JOB_ID
echo Running on nodes: $SLURM_JOB_NODELIST
echo Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE
echo Memory per node: $SLURM_MEM_PER_NODE MB
echo Job name: $SLURM_JOB_NAME
/home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_cv3_mcc_fix_plots_and_scaler_and_stratify/logs/config_0.sh
scontrol show job $SLURM_JOB_ID | grep -m 1 "RunTime=" >> /home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_cv3_mcc_fix_plots_and_scaler_and_stratify/logs/$(echo $SLURM_JOB_NAME)_$(echo $SLURM_JOB_ID)_log.txt
scontrol show job $SLURM_JOB_ID | grep -m 1 "NumCPUs=" >> /home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_cv3_mcc_fix_plots_and_scaler_and_stratify/logs/$(echo $SLURM_JOB_NAME)_$(echo $SLURM_JOB_ID)_log.txt
