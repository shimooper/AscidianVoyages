import pandas as pd

from base_model import BaseModel


class TwoDayModel(BaseModel):
    def __init__(self, all_outputs_dir_path):
        super().__init__('two_day_model', all_outputs_dir_path)

    def convert_routes_to_model_data(self, df):
        temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))

        two_day_data = []
        for index, row in df.iterrows():
            for col in temperature_columns[-1:0:-1]:
                col_day = int(col.split(' ')[1])
                if pd.isna(row[f'Temp {col_day}']):
                    continue
                new_row = {
                    'current day temperature': row[f'Temp {col_day}'],
                    'previous day temperature': row[f'Temp {col_day - 1}'],
                    'current day salinity': row[f'Salinity {col_day}'],
                    'previous day salinity': row[f'Salinity {col_day - 1}'],
                    'death': row[f'Lived {col_day}']
                }
                two_day_data.append(new_row)

        two_day_df = pd.DataFrame(two_day_data)
        for col in two_day_df.columns:
            two_day_df[col] = two_day_df[col].astype(int)

        return two_day_df
