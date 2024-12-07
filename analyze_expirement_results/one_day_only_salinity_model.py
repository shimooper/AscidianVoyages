import pandas as pd

from base_model import BaseModel


class OneDayOnlySalinityModel(BaseModel):
    def __init__(self, all_outputs_dir_path):
        super().__init__('one_day_only_salinity_model', all_outputs_dir_path)

    def convert_routes_to_model_data(self, df):
        lived_columns = sorted([col for col in df.columns if 'Lived' in col], key=lambda x: int(x.split()[1]))
        salinity_columns = sorted([col for col in df.columns if 'Salinity' in col], key=lambda x: int(x.split()[1]))

        one_day_salinity_data = []
        for idx, row in df.iterrows():
            for lived_col, salinity_col in zip(lived_columns, salinity_columns):
                if pd.isna(row[lived_col]):  # When we reached the end of route, continue to next route
                    break

                one_day_salinity_data.append({
                    'salinity': row[salinity_col],
                    'death': row[lived_col],
                })

        one_day_salinity_df = pd.DataFrame(one_day_salinity_data)
        for col in one_day_salinity_df.columns:
            one_day_salinity_df[col] = one_day_salinity_df[col].astype(int)

        return one_day_salinity_df
