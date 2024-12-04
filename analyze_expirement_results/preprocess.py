import pandas as pd

from utils import DATA_PATH, DATA_PROCESSED_PATH, variable_equals_value


def main():
    df = pd.read_excel(DATA_PATH, sheet_name='final_data')

    # Convert Temp, Salinity, Lived columns to integers
    df.replace("\\", pd.NA, inplace=True)
    lived_columns = sorted([col for col in df.columns if 'Lived' in col], key=lambda x: int(x.split()[1]))
    temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))
    salinity_columns = sorted([col for col in df.columns if 'Salinity' in col], key=lambda x: int(x.split()[1]))

    for col in lived_columns + temperature_columns + salinity_columns + ['Suspected from time point']:
        df[col] = pd.to_numeric(df[col], errors='raise').astype("Int64")

    # In Lived columns: replace 1 with 0, 0 with 1, keeping NaN as is
    for col in lived_columns:
        df[col] = df[col].apply(lambda x: 1 if variable_equals_value(x, 0) else (0 if variable_equals_value(x, 1) else x))

    # Convert all values in Lived columns to integers again
    for col in lived_columns:
        df[col] = pd.to_numeric(df[col], errors='raise').astype("Int64")

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

    df.to_csv(DATA_PROCESSED_PATH, index=False)


if __name__ == '__main__':
    main()
