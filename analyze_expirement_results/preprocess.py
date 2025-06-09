import re
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import variable_equals_value, get_column_groups_sorted
from configuration import DATA_PATH, Config, PROCESSED_DATA_DIR, PROCESSED_DATA_PATH


def preprocess_data(logger):
    if PROCESSED_DATA_PATH.exists():
        logger.info(f"Preprocessed data already exists at {PROCESSED_DATA_PATH}.")
        return

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_PATH, sheet_name='final_data')
    logger.info(f"Loaded data from {DATA_PATH}.")

    cols_to_drop = [col for col in df.columns if (('Lived' in col or 'Temp' in col or 'Salinity' in col)
                                                  and int(col.split()[1]) > 30)]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped columns: {cols_to_drop}.")

    df.replace("\\", pd.NA, inplace=True)
    logger.info(f"Replaced all backslashes with NaN values.")

    add_acclimation_days(df)
    logger.info(f"Added acclimation days columns.")
    df.columns = [shift_column_name(col) for col in df.columns]
    logger.info(f"Shifted column names to start with day 0")

    # Convert Temp, Salinity, Lived columns to integers
    lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)
    for col in lived_columns + temperature_columns + salinity_columns:
        df[col] = pd.to_numeric(df[col], errors='raise').astype("Int64")
    logger.info(f"Converted Temp, Salinity, Lived columns to integers.")

    replace_lived_indicators(df, lived_columns)
    logger.info(f"Replaced 1 with 0 and 0 with 1 in Lived columns.")
    add_dying_day(df, lived_columns, temperature_columns, salinity_columns)
    logger.info(f"Added dying_day column.")

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Preprocessed data saved to {PROCESSED_DATA_PATH}\n{routes_statistics(df)}")


def preprocess_data_by_config(logger, config: Config, routes_df):
    if not config.include_control_routes:
        routes_df = routes_df[routes_df['Name'] != 'CONTROL']

    if not config.include_suspected_routes:
        remove_suspected_routes_parts(routes_df)

    processed_routes_path = config.data_dir_path / 'full.csv'
    routes_df.to_csv(processed_routes_path, index=False)

    ship_names_and_seasons = routes_df[['Name', 'Season']].drop_duplicates()
    ship_names_and_seasons_no_control = ship_names_and_seasons[ship_names_and_seasons['Name'] != 'CONTROL']

    if config.stratify:
        stratify_column = ship_names_and_seasons_no_control['Season']
    else:
        stratify_column = None

    train_ship_names_seasons_df, test_ship_names_seasons_df = train_test_split(
        ship_names_and_seasons_no_control, test_size=config.test_set_size, random_state=config.random_state, stratify=stratify_column)

    # Add CONTROL routes to the train set
    control_names_seasons = pd.DataFrame({'Name': ['CONTROL', 'CONTROL'], 'Season': ['winter', 'summer']})
    train_ship_names_seasons_df = pd.concat([train_ship_names_seasons_df, control_names_seasons], ignore_index=True)

    train_df = routes_df.merge(train_ship_names_seasons_df, on=['Name', 'Season'], how='inner')
    test_df = routes_df.merge(test_ship_names_seasons_df, on=['Name', 'Season'], how='inner')

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
        row_lived_columns = row[lived_columns]
        last_col_of_survival = row_lived_columns[row_lived_columns == 0].index[-1]
        last_col_of_survival_index = lived_columns.index(last_col_of_survival)

        # Fill all previous columns with 0
        df.loc[idx, lived_columns[:last_col_of_survival_index + 1]] = 0

        first_col_of_death_index = last_col_of_survival_index + 1
        if first_col_of_death_index < len(lived_columns) and variable_equals_value(df.loc[idx, lived_columns[first_col_of_death_index]], 1):
            # Assign nan to all following columns
            df.loc[idx, lived_columns[first_col_of_death_index + 1:]] = pd.NA
            df.loc[idx, temperature_columns[first_col_of_death_index + 1:]] = pd.NA
            df.loc[idx, salinity_columns[first_col_of_death_index + 1:]] = pd.NA

            dying_day[idx] = int(lived_columns[first_col_of_death_index].split()[1])
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


def dying_day_is_suspected(row_dying_day, control_deaths):
    if pd.isna(row_dying_day):
        return False

    # Check if the dying day is within 2 days of any control death
    return any(int(row_dying_day) in range(int(control_death) - 2, int(control_death) + 3) for control_death in control_deaths)


def remove_suspected_routes_parts(df):
    lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

    suspected_control_routes_df = df[(df['Name'] == 'CONTROL') & (df['dying_day'].notna())]
    month_to_control_deaths = suspected_control_routes_df.groupby('Sampling Date')['dying_day'].apply(list).to_dict()

    for idx, row in df.iterrows():
        control_deaths = month_to_control_deaths.get(row['Sampling Date'], [])
        if not control_deaths:
            continue

        if dying_day_is_suspected(row['dying_day'], control_deaths):
            for col in lived_columns + temperature_columns + salinity_columns:
                if int(col.split()[1]) >= row['dying_day']:
                    df.loc[idx, col] = pd.NA

            df.loc[idx, 'dying_day'] = pd.NA


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


def shift_column_name(name):
    match = re.match(r'^(Lived|Temp|Salinity) (-?\d+)$', name)
    if match:
        prefix, number = match.groups()
        new_number = int(number) + 3
        return f"{prefix} {new_number}"
    return name  # keep unchanged if it doesn't match


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
