import pandas as pd

from base_model import BaseModel
from utils import get_column_groups_sorted, convert_columns_to_int


class OneWeekModel(BaseModel):
    def __init__(self, all_outputs_dir_path, metric_to_choose_best_model, random_state, model_id):
        super().__init__('one_week_model', all_outputs_dir_path, metric_to_choose_best_model, random_state, model_id)

    def convert_routes_to_model_data(self, df):
        lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

        one_week_data = []
        for index, row in df.iterrows():
            for col in temperature_columns[-1:5:-1]:
                col_day = int(col.split(' ')[1])
                if pd.isna(row[f'Temp {col_day}']):
                    continue
                week_temperature_column = [f'Temp {col_day - i}' for i in range(7)]
                week_salinity_column = [f'Salinity {col_day - i}' for i in range(7)]
                new_row = {
                    'current day temperature': row[f'Temp {col_day}'],
                    'previous day temperature': row[f'Temp {col_day - 1}'],
                    'current day salinity': row[f'Salinity {col_day}'],
                    'previous day salinity': row[f'Salinity {col_day - 1}'],
                    'weekly average temperature': row[week_temperature_column].mean(),
                    'weekly average salinity': row[week_salinity_column].mean(),
                    'weekly max temperature': row[week_temperature_column].max(),
                    'weekly min temperature': row[week_temperature_column].min(),
                    'weekly max salinity': row[week_salinity_column].max(),
                    'weekly min salinity': row[week_salinity_column].min(),
                    'death': row[f'Lived {col_day}']
                }
                one_week_data.append(new_row)

        one_week_df = pd.DataFrame(one_week_data)
        convert_columns_to_int(one_week_df, columns_to_skip=['weekly average temperature', 'weekly average salinity'])

        return one_week_df
