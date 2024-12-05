import pandas as pd

from base_model import BaseModel


class OneWeekModel(BaseModel):
    def __init__(self, train_file_path, test_file_path, all_outputs_dir_path):
        super().__init__('one_week_model', train_file_path, test_file_path, all_outputs_dir_path)

    def convert_routes_to_model_data(self, df):
        temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))

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

        for col in [col for col in one_week_df.columns if
                    col not in ['weekly average temperature', 'weekly average salinity']]:
            one_week_df[col] = one_week_df[col].astype(int)

        return one_week_df
