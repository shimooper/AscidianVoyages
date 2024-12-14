import pandas as pd

from base_model import BaseModel
from utils import get_column_groups_sorted, convert_columns_to_int


class OneDayOnlySalinityModel(BaseModel):
    def __init__(self, all_outputs_dir_path, metric_to_choose_best_model, random_state,
                 number_of_future_days_to_consider_death, model_id):
        super().__init__('one_day_only_salinity_model', all_outputs_dir_path, metric_to_choose_best_model, random_state,
                         number_of_future_days_to_consider_death, model_id)

    def convert_routes_to_model_data(self, df, number_of_future_days_to_consider_death):
        lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

        one_day_salinity_data = []
        for idx, row in df.iterrows():
            for lived_col, salinity_col in zip(lived_columns, salinity_columns):
                if pd.isna(row[lived_col]):  # When we reached the end of route, continue to next route
                    break

                col_day = int(lived_col.split(' ')[1])
                lived_cols_to_consider = [f'Lived {col_day + offset}' for offset in range(0, number_of_future_days_to_consider_death + 1)]
                one_day_salinity_data.append({
                    'salinity': row[salinity_col],
                    'death': any(row[lived_cols_to_consider]),
                })

        one_day_salinity_df = pd.DataFrame(one_day_salinity_data)
        convert_columns_to_int(one_day_salinity_df)

        return one_day_salinity_df
