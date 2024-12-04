import pandas as pd

from utils import DATA_PATH, DATA_PROCESSED_PATH


def variable_equals_value(variable, value):
    if pd.isna(value):
        return pd.isna(variable)

    return pd.notna(variable) and variable == value


def main():
    df = pd.read_excel(DATA_PATH, sheet_name='final_data')

    # Convert Temp, Salinity, Lived columns to integers
    df.replace("\\", pd.NA, inplace=True)
    lived_columns = sorted([col for col in df.columns if 'Lived' in col], key=lambda x: int(x.split()[1]))
    temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))
    salinity_columns = sorted([col for col in df.columns if 'Salinity' in col], key=lambda x: int(x.split()[1]))

    for col in lived_columns + temperature_columns + salinity_columns + ['Suspected from time point']:
        df[col] = pd.to_numeric(df[col], errors='raise').astype("Int64")

    # Add dying_day column and convert all values after death to nan
    dying_day = {}
    # Iterate over each row
    for idx, row in df.iterrows():
        # Find the last column containing 1
        last_col_with_one = None
        for col in lived_columns:
            if variable_equals_value(row[col], 1):
                last_col_with_one = col

        # Fill all previous columns with 1
        last_col_with_one_index = lived_columns.index(last_col_with_one)
        df.loc[idx, lived_columns[:last_col_with_one_index + 1]] = 1

        first_col_with_zero_index = last_col_with_one_index + 1
        if first_col_with_zero_index < len(lived_columns) and variable_equals_value(df.loc[idx, lived_columns[first_col_with_zero_index]], 0):
            # Assign nan to all following columns
            df.loc[idx, lived_columns[first_col_with_zero_index + 1:]] = pd.NA
            df.loc[idx, temperature_columns[first_col_with_zero_index + 1:]] = pd.NA
            df.loc[idx, salinity_columns[first_col_with_zero_index + 1:]] = pd.NA

            dying_day[idx] = first_col_with_zero_index
        else:  # The animal lived until the end of the experiment
            dying_day[idx] = pd.NA

    df['dying_day'] = df.index.map(dying_day)
    df.to_csv(DATA_PROCESSED_PATH, index=False)


if __name__ == '__main__':
    main()
