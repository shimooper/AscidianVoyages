import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from base_model import BaseModel
from utils import get_column_groups_sorted, convert_columns_to_int


class OneDayModel(BaseModel):
    def __init__(self, all_outputs_dir_path, metric_to_choose_best_model, random_state, model_id):
        super().__init__('one_day_model', all_outputs_dir_path, metric_to_choose_best_model, random_state, model_id)

    def convert_routes_to_model_data(self, df):
        lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

        one_day_data = []
        for idx, row in df.iterrows():
            for lived_col, temp_col, salinity_col in zip(lived_columns, temperature_columns, salinity_columns):
                if pd.isna(row[lived_col]):  # When we reached the end of route, continue to next route
                    break

                one_day_data.append({
                    'temperature': row[temp_col],
                    'salinity': row[salinity_col],
                    'death': row[lived_col],
                })

        one_day_df = pd.DataFrame(one_day_data)
        convert_columns_to_int(one_day_df)

        return one_day_df

    def plot_one_day_data(self):
        one_day_train_df = pd.read_csv(self.model_train_set_path)
        one_day_test_df = pd.read_csv(self.model_test_set_path)
        one_day_full_df = pd.concat([one_day_train_df, one_day_test_df], axis=0)

        one_day_full_df['death_label'] = one_day_full_df['death'].map({1: 'Dead', 0: 'Alive'})
        sns.scatterplot(x='temperature', y='salinity', hue='death_label', data=one_day_full_df, alpha=0.7, palette={'Dead': 'red', 'Alive': 'green'})
        plt.xlabel("Temperature (celsius)")
        plt.ylabel("Salinity (ppt)")
        plt.title("One Day Model")
        plt.legend(title="Status")
        plt.grid(alpha=0.3)
        plt.savefig(self.model_data_dir / "scatter_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

    def run_analysis(self):
        super().run_analysis()
        self.plot_one_day_data()
