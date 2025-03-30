import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import variable_equals_value, setup_logger, get_column_groups_sorted
from configuration import DATA_DIR, DATA_PATH, Config


def permanent_preprocess_data():
    outputs_preprocess_dir = DATA_DIR / 'preprocess'
    data_processed_path = outputs_preprocess_dir / 'Final_Data_Voyages_Processed.csv'

    if os.path.exists(data_processed_path):
        return pd.read_csv(data_processed_path)

    os.makedirs(outputs_preprocess_dir, exist_ok=True)
    logger = setup_logger(outputs_preprocess_dir / 'preprocess.log', 'PREPROCESS')

    df = pd.read_excel(DATA_PATH, sheet_name='final_data')
    logger.info(f"Loaded data from {DATA_PATH}.")

    df.replace("\\", pd.NA, inplace=True)
    logger.info(f"Replaced all backslashes with NaN values.")

    # Convert Temp, Salinity, Lived columns to integers
    lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)
    for col in lived_columns + temperature_columns + salinity_columns + ['Suspected from time point']:
        df[col] = pd.to_numeric(df[col], errors='raise').astype("Int64")
    logger.info(f"Converted Temp, Salinity, Lived columns to integers.")

    replace_lived_indicators(df, lived_columns)
    logger.info(f"Replaced 1 with 0 and 0 with 1 in Lived columns.")
    add_dying_day(df, lived_columns, temperature_columns, salinity_columns)
    logger.info(f"Added dying_day column.")
    add_acclimation_days(df)
    logger.info(f"Added acclimation days columns.")

    df.to_csv(data_processed_path, index=False)
    logger.info(f"Preprocessed data saved to {data_processed_path}\n{routes_statistics(df)}")

    return df


def preprocess_data_by_config(config: Config, routes_df):
    logger = setup_logger(config.data_dir_path / 'preprocess.log', f'CONFIGURATION_{config.configuration_id}_PREPROCESS')

    if not config.include_control_routes:
        routes_df = routes_df[routes_df['Name'] != 'CONTROL']

    if not config.include_suspected_routes:
        remove_suspected_routes_parts(routes_df)

    processed_routes_path = config.data_dir_path / 'full.csv'
    routes_df.to_csv(processed_routes_path, index=False)

    ship_names_and_seasons = routes_df[['Name', 'Season']].drop_duplicates()
    ship_names_and_seasons_no_control = ship_names_and_seasons[ship_names_and_seasons['Name'] != 'CONTROL']

    if config.stratify:
        # routes_df['stratify_group'] = routes_df['Season'] + '_' + routes_df['dying_day'].notna().astype(int).astype(str)
        # stratify_column = routes_df['stratify_group']
        stratify_column = routes_df['Season']
    else:
        stratify_column = None

    train_ship_names_df, test_ship_names_df = train_test_split(ship_names_and_seasons_no_control, test_size=config.test_set_size,
                                                               random_state=config.random_state, stratify=stratify_column)
    train_df = routes_df[routes_df['Name'].isin(list(train_ship_names_df['Name']) + ['CONTROL'])]
    test_df = routes_df[routes_df['Name'].isin(test_ship_names_df['Name'])]

    train_df.to_csv(config.data_dir_path / 'train.csv', index=False)
    test_df.to_csv(config.data_dir_path / 'test.csv', index=False)

    logger.info(f"Applied filters: include_control_routes={config.include_control_routes}, "
                f"include_suspected_routes={config.include_suspected_routes}, stratify={config.stratify}\n"
                f"{routes_statistics(routes_df)}\n"
                f"Routes were saved to {processed_routes_path}.\n"
                f"There are {len(train_df)} routes in the train set and {len(test_df)} routes in the test set.")


def replace_lived_indicators(df, lived_columns):
    # In Lived columns: replace 1 with 0, 0 with 1, keeping NaN as is
    for col in lived_columns:
        df[col] = df[col].apply(lambda x: 1 if variable_equals_value(x, 0) else (0 if variable_equals_value(x, 1) else x))

    # Convert all values in Lived columns to integers again
    for col in lived_columns:
        df[col] = pd.to_numeric(df[col], errors='raise').astype("Int64")


def add_dying_day(df, lived_columns, temperature_columns, salinity_columns):
    # Add dying_day column and convert all values after death to nan
    dying_day = {}
    # Iterate over each row
    for idx, row in df.iterrows():
        # Find the last column of survival (containing 0)
        last_col_of_survival = None
        for col in lived_columns:
            if variable_equals_value(row[col], 0):
                last_col_of_survival = col

        # Fill all previous columns with 0
        last_col_of_survival_index = lived_columns.index(last_col_of_survival)
        df.loc[idx, lived_columns[:last_col_of_survival_index + 1]] = 0

        first_col_of_death_index = last_col_of_survival_index + 1
        if first_col_of_death_index < len(lived_columns) and variable_equals_value(df.loc[idx, lived_columns[first_col_of_death_index]], 1):
            # Assign nan to all following columns
            df.loc[idx, lived_columns[first_col_of_death_index + 1:]] = pd.NA
            df.loc[idx, temperature_columns[first_col_of_death_index + 1:]] = pd.NA
            df.loc[idx, salinity_columns[first_col_of_death_index + 1:]] = pd.NA

            dying_day[idx] = first_col_of_death_index
        else:  # The animal lived until the end of the experiment
            dying_day[idx] = pd.NA

    df['dying_day'] = df.index.map(dying_day)

    # validate that the nan values are consistent in all columns
    for idx, row in df.iterrows():
        for lived_col, temp_col, salinity_col in zip(lived_columns, temperature_columns, salinity_columns):
            if pd.isna(row[lived_col]) or pd.isna(row[temp_col]) or pd.isna(row[salinity_col]):
                if pd.isna(row[lived_col]) and pd.isna(row[temp_col]) and pd.isna(row[salinity_col]):
                    continue
                else:
                    raise ValueError(f'Row {idx} contains NaN values in some but not all of the columns: '
                                     f'{lived_col}, {temp_col}, {salinity_col}')


def remove_suspected_routes_parts(df):
    lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

    for idx, row in df.iterrows():
        if pd.isna(row['Suspected from time point']):
            continue

        if row['Suspected from time point'] != row['dying_day']:
            raise ValueError(f"Row {idx} has different values in 'Suspected from time point' and 'dying_day' columns.")

        for col in lived_columns + temperature_columns + salinity_columns:
            if int(col.split()[1]) >= row['Suspected from time point']:
                df.loc[idx, col] = pd.NA


def add_acclimation_days(df):
    # Add columns for the 2 days before Day 0 because there were total 3 days of acclimation
    temp_0_col_index = df.columns.get_loc('Temp 0')
    df.insert(temp_0_col_index, 'Temp -1', df['Temp 0'])
    df.insert(temp_0_col_index, 'Temp -2', df['Temp 0'])

    salinity_0_col_index = df.columns.get_loc('Salinity 0')
    df.insert(salinity_0_col_index, 'Salinity -1', df['Salinity 0'])
    df.insert(salinity_0_col_index, 'Salinity -2', df['Salinity 0'])

    lived_0_col_index = df.columns.get_loc('Lived 0')
    df.insert(lived_0_col_index, 'Lived -1', df['Lived 0'])
    df.insert(lived_0_col_index, 'Lived -2', df['Lived 0'])


def routes_statistics(df):
    count = len(df)
    count_death = df['dying_day'].notna().sum()
    count_survived = df['dying_day'].isna().sum()

    routes_winter_df = df[df['Season'] == 'winter']
    count_winter = len(routes_winter_df)
    count_winter_death = routes_winter_df['dying_day'].notna().sum()
    count_winter_survived = routes_winter_df['dying_day'].isna().sum()

    routes_summer_df = df[df['Season'] == 'summer']
    count_summer = len(routes_summer_df)
    count_summer_death = routes_summer_df['dying_day'].notna().sum()
    count_summer_survived = routes_summer_df['dying_day'].isna().sum()

    text = f"There are {count} routes in the dataset. {count_death} of them ended with death, and {count_survived} did not.\n" \
           f"{count_winter} routes are in winter season, {count_winter_death} of them ended with death, and {count_winter_survived} did not.\n" \
           f"{count_summer} routes are in summer season, {count_summer_death} of them ended with death, and {count_summer_survived} did not."

    return text
