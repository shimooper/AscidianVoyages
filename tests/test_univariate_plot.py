import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\repos\GoogleShips\analyze_expirement_results\outputs\configuration_0\models\days_to_consider_4\data\train.csv")

df['death_label'] = df['death'].map({1: 'Dead', 0: 'Alive'}).astype('category')
df.drop(columns=['death'], inplace=True)

# Define your column groups
temp_cols = [col for col in df.columns if 'temp' in col.lower()]
salinity_cols = [col for col in df.columns if 'salinity' in col.lower()]

# Melt dataframes for plotting
df_temp = df[temp_cols + ['death_label']].melt(id_vars='death_label', var_name='feature', value_name='value')
df_salinity = df[salinity_cols + ['death_label']].melt(id_vars='death_label', var_name='feature', value_name='value')

# Set up side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Temperature plot
sns.stripplot(data=df_temp, x='feature', y='value', hue='death_label',
              jitter=True, alpha=0.7, ax=axes[0], legend=False)
axes[0].set_ylabel("Temperature (celsius)", fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_xlabel(None)

# Salinity plot
sns.stripplot(data=df_salinity, x='feature', y='value', hue='death_label',
              jitter=True, alpha=0.7, ax=axes[1])
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set_ylabel("Salinity (ppt)", fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_xlabel(None)
axes[1].legend(title=None)

# Global title
fig.suptitle("Feature distributions colored by label", fontsize=16)
plt.tight_layout()
plt.savefig("test_univariate_plot.png", dpi=600, bbox_inches='tight')
