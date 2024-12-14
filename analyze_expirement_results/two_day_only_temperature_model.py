import pandas as pd

from base_model import BaseModel
from utils import get_column_groups_sorted, convert_columns_to_int, get_lived_columns_to_consider


class TwoDayOnlyTemperatureModel(BaseModel):
    def __init__(self, all_outputs_dir_path, metric_to_choose_best_model, random_state,
                 number_of_future_days_to_consider_death, model_id):
        super().__init__('two_day_only_temperature_model', all_outputs_dir_path, metric_to_choose_best_model, random_state,
                         number_of_future_days_to_consider_death, model_id)

    def convert_routes_to_model_data(self, df, number_of_future_days_to_consider_death):
        lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

        two_day_data = []
        for index, row in df.iterrows():
            for col in lived_columns[-1:0:-1]:
                col_day = int(col.split(' ')[1])
                if pd.isna(row[f'Lived {col_day}']):
                    continue

                lived_cols_to_consider = get_lived_columns_to_consider(row, col_day, number_of_future_days_to_consider_death)
                new_row = {
                    'current day temperature': row[f'Temp {col_day}'],
                    'previous day temperature': row[f'Temp {col_day - 1}'],
                    'death': any(row[lived_cols_to_consider]),
                }
                two_day_data.append(new_row)

        two_day_df = pd.DataFrame(two_day_data)
        convert_columns_to_int(two_day_df)

        return two_day_df
