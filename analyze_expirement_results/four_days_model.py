import pandas as pd

from base_model import BaseModel
from utils import get_column_groups_sorted, convert_columns_to_int, get_lived_columns_to_consider


class FourDaysModel(BaseModel):
    def __init__(self, all_outputs_dir_path, metric_to_choose_best_model, random_state,
                 number_of_future_days_to_consider_death, model_id):
        super().__init__('four_days_model', all_outputs_dir_path, metric_to_choose_best_model, random_state,
                         number_of_future_days_to_consider_death, model_id)

    def convert_routes_to_model_data(self, df, number_of_future_days_to_consider_death):
        lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

        four_days_data = []
        for index, row in df.iterrows():
            for col in lived_columns[-1:2:-1]:
                col_day = int(col.split(' ')[1])
                if pd.isna(row[f'Lived {col_day}']):
                    continue

                temperature_columns = [f'Temp {col_day - i}' for i in range(4)]
                salinity_columns = [f'Salinity {col_day - i}' for i in range(4)]
                lived_cols_to_consider = get_lived_columns_to_consider(row, col_day, number_of_future_days_to_consider_death)
                new_row = {
                    'current day temperature': row[f'Temp {col_day}'],
                    'previous day temperature': row[f'Temp {col_day - 1}'],
                    '2 days ago temperature': row[f'Temp {col_day - 2}'],
                    '3 days ago temperature': row[f'Temp {col_day - 3}'],
                    'max temperature': row[temperature_columns].max(),
                    'min temperature': row[temperature_columns].min(),
                    'current day salinity': row[f'Salinity {col_day}'],
                    'previous day salinity': row[f'Salinity {col_day - 1}'],
                    '2 days ago salinity': row[f'Salinity {col_day - 2}'],
                    '3 days ago salinity': row[f'Salinity {col_day - 3}'],
                    'max salinity': row[salinity_columns].max(),
                    'min salinity': row[salinity_columns].min(),
                    'death': any(row[lived_cols_to_consider]),
                }
                four_days_data.append(new_row)

        four_days_df = pd.DataFrame(four_days_data)
        convert_columns_to_int(four_days_df)

        return four_days_df
