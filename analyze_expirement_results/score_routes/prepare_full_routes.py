from pathlib import Path
import pandas as pd
import argparse
import re


def find_first_nan_day(row: pd.Series):
    for col in row.index:
        if 'Temp' in col:
            value = row[col]
            if pd.isna(value) or value == '':
                # Extract the number using regular expressions
                match = re.search(r'Temp\s*([-]?\d+)', col)
                if match:
                    return int(match.group(1))
                else:
                    raise ValueError(f"Could not extract day number from column name: {col}")

    return None


def prepare_data(actual_experiment_data, planned_experiment_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_df = pd.read_csv(actual_experiment_data)
    planned_df = pd.read_csv(planned_experiment_data)

    for actual_df_index, row in actual_df.iterrows():
        try:
            first_nan_day = find_first_nan_day(row)
            if first_nan_day is not None:
                if row['Name'] == 'CONTROL':
                    planned_temp = lambda day_number: row['Temp 0']
                    planned_salinity = lambda day_number: row['Salinity 0']
                else:
                    planned_route = planned_df[(planned_df['Ship'].str.contains(row['Name'])) &
                                               (planned_df['Season'] == row['Season'])].reset_index()
                    planned_temp = lambda day_number: round(planned_route.loc[day_number - 1, 'Temperature (celsius)'])
                    planned_salinity = lambda day_number: round(planned_route.loc[day_number - 1, 'Salinity (ppt)'])

                for i in range(first_nan_day, 30 + 1):
                    actual_df.loc[actual_df_index, f'Temp {i}'] = planned_temp(i)
                    actual_df.loc[actual_df_index, f'Salinity {i}'] = planned_salinity(i)
        except Exception as e:
            print(f"Error processing row {actual_df_index}: {e}")
            continue

    actual_df.to_csv(output_dir / f'actual_routes_augmented_{actual_experiment_data.stem}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for scoring routes')
    parser.add_argument('--actual_experiment_data', type=Path,
                        help='Path to the actual experiment data', default=Path('actual_routes') / 'test.csv')
    parser.add_argument('--planned_experiment_data', type=Path,
                        help='Path to the planned experiment data', default=Path('planned_routes') / 'all_sampled_routes.csv')
    parser.add_argument('--output_dir', type=Path,
                        help='Directory to save the prepared data', default=Path('full_routes'))
    args = parser.parse_args()

    prepare_data(args.actual_experiment_data, args.planned_experiment_data, args.output_dir)
