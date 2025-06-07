from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


BEST_CLASSIFIERS_DF_PATH = Path(r"C:\repos\GoogleShips\analyze_expirement_results\outputs_test_local\configuration_0\models\days_to_consider_4\train_outputs\best_classifier\best_classifier_from_each_class.csv")
outputs_dir = Path('.')
metric = 'f1'

def main():
    results_df = pd.read_csv(BEST_CLASSIFIERS_DF_PATH, index_col=0)

    results_filtered_df = results_df[['mean_test_f1', 'mean_test_auprc', 'mean_test_mcc']].copy()
    results_filtered_df.rename(columns={'mean_test_f1': 'F1', 'mean_test_auprc': 'AUPRC', 'mean_test_mcc': 'MCC'}, inplace=True)
    results_filtered_melted_df = results_filtered_df.reset_index().melt(id_vars='model_name', var_name='metric', value_name='score')

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", context="paper")
    palette = sns.color_palette("colorblind", n_colors=len(results_filtered_df.index))

    sns.barplot(data=results_filtered_melted_df, x='metric', y='score', hue='model_name', palette=palette)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Metric', fontsize=14)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., title_fontsize=14, fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(outputs_dir / 'models_comparison.png', dpi=600)
    plt.close()

    max_indices = results_filtered_df.idxmax()
    results_filtered_df.loc['best_model'] = max_indices
    results_filtered_df.to_csv(outputs_dir / 'models_comparison.csv')

    best_model_name = results_df[f'mean_test_{metric}'].idxmax()
    best_model_fold_scores = results_df.loc[best_model_name, [f'split0_test_{metric}', f'split1_test_{metric}',
                                                              f'split2_test_{metric}', f'split3_test_{metric}',
                                                              f'split4_test_{metric}']].values
    statistics_results = []
    for model in results_df.index:
        if model == best_model_name:
            continue
        other_f1_scores = results_df.loc[
            model, ['split0_test_f1', 'split1_test_f1', 'split2_test_f1', 'split3_test_f1', 'split4_test_f1']].values

        # Paired t-test
        t_stat, p_val_ttest = ttest_rel(best_model_fold_scores, other_f1_scores)

        # Wilcoxon signed-rank test
        try:
            stat_wilcox, p_val_wilcox, _ = wilcoxon(best_model_fold_scores, other_f1_scores)
        except ValueError:
            p_val_wilcox = 1.0  # fallback if no difference

        statistics_results.append({
            'model_compared': model,
            'p_val_ttest': p_val_ttest,
            'p_val_wilcox': p_val_wilcox,
            'mean_diff': best_model_fold_scores.mean() - other_f1_scores.mean()
        })

    statistics_results_df = pd.DataFrame(statistics_results)

    # FDR correction for t-test p-values
    reject_ttest, pvals_ttest_fdr, _, _ = multipletests(statistics_results_df['p_val_ttest'], method='fdr_bh')
    statistics_results_df['p_val_ttest_fdr_corrected'] = pvals_ttest_fdr
    statistics_results_df['significant_ttest_fdr'] = reject_ttest

    # FDR correction for Wilcoxon p-values
    reject_wilcox, pvals_wilcox_fdr, _, _ = multipletests(statistics_results_df['p_val_wilcox'], method='fdr_bh')
    statistics_results_df['p_val_wilcox_fdr_corrected'] = pvals_wilcox_fdr
    statistics_results_df['significant_wilcox_fdr'] = reject_wilcox

    statistics_results_df.to_csv(outputs_dir / 'model_comparison_statistics.csv', index=False)


if __name__ == "__main__":
    main()
