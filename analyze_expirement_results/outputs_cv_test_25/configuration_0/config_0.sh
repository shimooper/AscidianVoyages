source /home/ai_center/ai_users/yairshimony/miniconda/etc/profile.d/conda.sh
conda activate /home/ai_center/ai_users/yairshimony/miniconda/envs/ships
export PATH=$CONDA_PREFIX/bin:$PATH
python /home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/run_analysis_of_config.py /home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_cv3_mcc_fix_plots_and_scaler_and_stratify/configuration_0/config.csv --logs_dir /home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_cv3_mcc_fix_plots_and_scaler_and_stratify/logs --error_file_path /home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_cv3_mcc_fix_plots_and_scaler_and_stratify/error.txt --job_name config_0 --cpus 8
touch /home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_cv3_mcc_fix_plots_and_scaler_and_stratify/logs/config_0.done
