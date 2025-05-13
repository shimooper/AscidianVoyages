import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv(r"C:\repos\GoogleShips\analyze_expirement_results\outputs\configuration_0\models\days_to_consider_4\data\train.csv")
test_df = pd.read_csv(r"C:\repos\GoogleShips\analyze_expirement_results\outputs\configuration_0\models\days_to_consider_4\data\test.csv")

full_df = pd.concat([train_df, test_df], axis=0)

full_df['death_label'] = full_df['death'].map({1: 'Dead', 0: 'Alive'}).astype('category')
full_df.drop(columns=['death'], inplace=True)

temp_cols = [col for col in full_df.columns if 'temp' in col.lower()]
salinity_cols = [col for col in full_df.columns if 'salinity' in col.lower()]

df_temp = full_df[temp_cols + ['death_label']].melt(id_vars='death_label', var_name='feature',
                                                    value_name='value')
df_salinity = full_df[salinity_cols + ['death_label']].melt(id_vars='death_label', var_name='feature',
                                                            value_name='value')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.stripplot(data=df_temp, x='feature', y='value', hue='death_label', jitter=True, alpha=0.7, ax=axes[0],
              legend=False, palette={'Alive': 'green', 'Dead': 'red'})
axes[0].set_ylabel("Temperature (celsius)", fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_xlabel(None)

sns.stripplot(data=df_salinity, x='feature', y='value', hue='death_label', jitter=True, alpha=0.7, ax=axes[1],
              palette={'Alive': 'green', 'Dead': 'red'})
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set_ylabel("Salinity (ppt)", fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_xlabel(None)
axes[1].legend(title=None)

fig.suptitle("Feature distributions colored by label", fontsize=16)
plt.savefig("scatter_plot.png", dpi=600, bbox_inches='tight')
plt.close()
