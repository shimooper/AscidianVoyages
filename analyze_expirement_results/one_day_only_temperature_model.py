import pandas as pd

from base_model import BaseModel


class OneDayOnlyTemperatureModel(BaseModel):
    def __init__(self, all_outputs_dir_path, metric_to_choose_best_model, random_state, model_id):
        super().__init__('one_day_only_temperature_model', all_outputs_dir_path, metric_to_choose_best_model, random_state, model_id)

    def convert_routes_to_model_data(self, df):
        lived_columns = sorted([col for col in df.columns if 'Lived' in col], key=lambda x: int(x.split()[1]))
        temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))

        one_day_only_temperature_data = []
        for idx, row in df.iterrows():
            for lived_col, temp_col in zip(lived_columns, temperature_columns):
                if pd.isna(row[lived_col]):  # When we reached the end of route, continue to next route
                    break

                one_day_only_temperature_data.append({
                    'temperature': row[temp_col],
                    'death': row[lived_col],
                })

        one_day_only_temperature_df = pd.DataFrame(one_day_only_temperature_data)
        for col in one_day_only_temperature_df.columns:
            one_day_only_temperature_df[col] = one_day_only_temperature_df[col].astype(int)

        return one_day_only_temperature_df
