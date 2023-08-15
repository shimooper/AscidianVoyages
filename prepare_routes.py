import os
import calendar
import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUTS_DIR = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\outputs\long_routes_dataset"

SHIPS_ROUTES_DATASET_EXTENDED = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\ships routes\ships routes (ports only) with locations of ports - extended.csv"
SHIPS_ROUTES_DATASET = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\ships routes\ships routes (ports only) with locations of ports.csv"

MONTHS_NAMES = calendar.month_name[1:]

TEMPRATURE_JANUARY_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_01.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_01.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_01.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_01.CSV"
]
TEMPRATURE_FEBRUARY_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_02.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_02.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_02.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_02.CSV"
]
TEMPRATURE_MARCH_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_03.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_03.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_03.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_03.CSV"
]
TEMPRATURE_APRIL_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_04.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_04.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_04.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_04.CSV"
]
TEMPRATURE_MAY_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_05.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_05.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_05.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_05.CSV"
]
TEMPRATURE_JUNE_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_06.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_06.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_06.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_06.CSV"
]
TEMPRATURE_JULY_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_07.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_07.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_07.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_07.CSV"
]
TEMPRATURE_AUGUST_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_08.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_08.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_08.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_08.CSV"
]
TEMPRATURE_SEPTEMBER_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_09.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_09.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_09.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_09.CSV"
]
TEMPRATURE_OCTOBER_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_10.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_10.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_10.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_10.CSV"
]
TEMPRATURE_NOVEMBER_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_11.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_11.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_11.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_11.CSV"
]
TEMPRATURE_DECEMBER_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2022\temprature_NASA_2022_12.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2021\temprature_NASA_2021_12.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2020\temprature_NASA_2020_12.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_12.CSV"
]

CHLOROPHYL_JANUARY_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_01.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_01.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_01.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_01.CSV"
]
CHLOROPHYL_FEBRUARY_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_02.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_02.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_02.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_02.CSV"
]
CHLOROPHYL_MARCH_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_03.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_03.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_03.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_03.CSV"
]
CHLOROPHYL_APRIL_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_04.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_04.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_04.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_04.CSV"
]
CHLOROPHYL_MAY_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_05.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_05.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_05.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_05.CSV"
]
CHLOROPHYL_JUNE_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_06.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_06.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_06.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_06.CSV"
]
CHLOROPHYL_JULY_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_07.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_07.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_07.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_07.CSV"
]
CHLOROPHYL_AUGUST_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_08.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_08.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_08.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_08.CSV"
]
CHLOROPHYL_SEPTEMBER_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_09.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_09.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_09.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_09.CSV"
]
CHLOROPHYL_OCTOBER_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_10.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_10.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_10.CSV"
]
CHLOROPHYL_NOVEMBER_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_11.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_11.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_11.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_11.CSV"
]
CHLOROPHYL_DECEMBER_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2022\chlorophyl_NASA_2022_12.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2021\chlorophyl_NASA_2021_12.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2020\chlorophyl_NASA_2020_12.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_12.CSV"
]

TEMPRATURE_PATHS = [
    TEMPRATURE_JANUARY_PATHS, TEMPRATURE_FEBRUARY_PATHS, TEMPRATURE_MARCH_PATHS, TEMPRATURE_APRIL_PATHS,
    TEMPRATURE_MAY_PATHS, TEMPRATURE_JUNE_PATHS, TEMPRATURE_JULY_PATHS, TEMPRATURE_AUGUST_PATHS,
    TEMPRATURE_SEPTEMBER_PATHS, TEMPRATURE_OCTOBER_PATHS, TEMPRATURE_NOVEMBER_PATHS, TEMPRATURE_DECEMBER_PATHS
]
CHLOROPHYL_PATHS = [
    CHLOROPHYL_JANUARY_PATHS, CHLOROPHYL_FEBRUARY_PATHS, CHLOROPHYL_MARCH_PATHS, CHLOROPHYL_APRIL_PATHS,
    CHLOROPHYL_MAY_PATHS, CHLOROPHYL_JUNE_PATHS, CHLOROPHYL_JULY_PATHS, CHLOROPHYL_AUGUST_PATHS,
    CHLOROPHYL_SEPTEMBER_PATHS, CHLOROPHYL_OCTOBER_PATHS, CHLOROPHYL_NOVEMBER_PATHS, CHLOROPHYL_DECEMBER_PATHS
]

# Extract long and lat keys from one dataset for example
temprature_2022_january_df = pd.read_csv(TEMPRATURE_JANUARY_PATHS[0])
LAT_KEYS = temprature_2022_january_df['lat/lon'].to_numpy(dtype=float).round(2)
LONG_KEYS = temprature_2022_january_df.columns[1:].to_numpy(dtype=float).round(2)

SALINITY_CEDA_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_01.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_02.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_03.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_04.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_05.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_06.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_07.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_08.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_09.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_10.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_11.csv",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_12.csv"
]

SALINITY_WOA_PATH = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salnity data from WOA\woa18_decav_s00mn04.csv"

PORTS_CONDITIONS_2011_PATH = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\Ports dataset with their temprature and salinity.csv"


def process_routes_dataset(routes_path):
    routes_df = pd.read_csv(routes_path)

    # Convert columns to right data types
    routes_df['Arrival (LT)'] = pd.to_datetime(
        routes_df['Arrival (LT)'], errors='coerce', dayfirst=True)
    routes_df['Departure (LT)'] = pd.to_datetime(
        routes_df['Departure (LT)'], errors='coerce', dayfirst=True)
    routes_df['Longitude'] = pd.to_numeric(
        routes_df['Longitude'], errors='coerce')
    routes_df['Latitude'] = pd.to_numeric(
        routes_df['Latitude'], errors='coerce')
    routes_df['Hours'] = pd.to_numeric(routes_df['Hours'], errors='coerce')

    mean_time_in_port_by_ship = routes_df.groupby(
        'Ship').mean(numeric_only=True)['Hours']

    final_ships_routes_dfs = []
    for name, ship_df in routes_df.groupby('Ship'):
        # Impute missing Hours (time in port) values
        ship_df.loc[ship_df['Hours'].isna(
        ), 'Hours'] = mean_time_in_port_by_ship[name]
        ship_df['Hours (time delta)'] = pd.to_timedelta(
            ship_df['Hours'].astype(str) + 'h', errors='coerce')

        # Impute missing Departure and Arrival values based on the imputed Hours values
        ship_df.loc[ship_df['Departure (LT)'].isna(
        ), 'Departure (LT)'] = ship_df['Arrival (LT)'] + ship_df['Hours (time delta)']
        ship_df.loc[ship_df['Arrival (LT)'].isna(
        ), 'Arrival (LT)'] = ship_df['Departure (LT)'] - ship_df['Hours (time delta)']

        # Prepare final route of the ship
        duplicated_ship_df = ship_df.copy()
        ship_df = ship_df[['Ship', 'Port', 'Arrival (LT)']]
        new_ship_df = ship_df.rename(columns={'Arrival (LT)': 'time'})
        duplicated_ship_df = duplicated_ship_df[[
            'Ship', 'Port', 'Departure (LT)']]
        duplicated_ship_df.rename(
            columns={'Departure (LT)': 'time'}, inplace=True)
        final_ship_df = pd.concat(
            [new_ship_df, duplicated_ship_df], ignore_index=True)
        final_ship_df.sort_values('time', ascending=True, inplace=True)

        # Convert (absolute) time column to relative time (in hours)
        reference_time_value = final_ship_df.iloc[0]['time']
        relative_times_in_hours = []
        for index, row in final_ship_df.iterrows():
            relative_times_in_hours.append(
                (row['time'] - reference_time_value) / np.timedelta64(1, 'h'))
        final_ship_df['relative time'] = relative_times_in_hours
        final_ships_routes_dfs.append(final_ship_df)

    final_routes_df = pd.concat(final_ships_routes_dfs, ignore_index=True)
    final_routes_df.to_csv(os.path.join(
        OUTPUTS_DIR, "routes_processed.csv"), index=False, date_format="%d/%m/%y %H:%M:%S")
    print(f"There are {len(final_routes_df.groupby('Ship'))} ships in the routes dataset, "
          f"with a mean route length of {final_routes_df.groupby('Ship').count()['Port'].mean() / 2} ports")

    return final_routes_df


def extract_ports_locations_df_from_routes_df(routes_path):
    routes_df = pd.read_csv(routes_path)
    ports_in_routes_df = routes_df[['Port', 'Longitude', 'Latitude']]
    ports_in_routes_df = ports_in_routes_df.drop_duplicates()
    ports_in_routes_df['Port'] = ports_in_routes_df['Port'].str.lower()

    print(f'There are {len(ports_in_routes_df)} ports in the routes dataset')
    return ports_in_routes_df


def prepare_temprature_or_chlorophyl_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df.columns = ['lat'] + list(LONG_KEYS)
    df['lat'] = LAT_KEYS
    return df


def read_temprature_and_chlorophyl_datasets():
    temprature_dfs = []
    for month_datasets_paths in TEMPRATURE_PATHS:
        month_dfs = [prepare_temprature_or_chlorophyl_dataset(dataset_path) for dataset_path in month_datasets_paths]
        temprature_dfs.append(month_dfs)

    chlorophyl_dfs = []
    for month_datasets_paths in CHLOROPHYL_PATHS:
        month_dfs = [prepare_temprature_or_chlorophyl_dataset(dataset_path) for dataset_path in month_datasets_paths]
        chlorophyl_dfs.append(month_dfs)

    return temprature_dfs, chlorophyl_dfs


def get_temprature_or_chlorophyl_value(datasets, lat, long):
    example_dataset = datasets[0]
    lat_index = example_dataset.index[example_dataset['lat'] == lat]
    long_index = example_dataset.columns.get_loc(long)

    coordinates_to_check = [(lat_index, long_index)]
    for i in range(10):
        coordinates_to_check.extend([(lat_index - i, long_index), (lat_index + i, long_index),
                                    (lat_index, long_index - i), (lat_index, long_index + i),
                                    (lat_index - i, long_index - i), (lat_index - i, long_index + i),
                                    (lat_index + i, long_index - i), (lat_index + i, long_index + i)])

    for lat_index, long_index in coordinates_to_check:
        for dataset in datasets:
            value = dataset.iloc[lat_index, long_index].item()
            if value != 99999:
                return value
    return None


def add_nasa_conditions_to_ports_df(ports_df):
    temprature_dfs, chlorophyl_dfs = read_temprature_and_chlorophyl_datasets()

    ports_temprature_per_month = [[] for _ in range(12)]
    ports_chlorophyl_per_month = [[] for _ in range(12)]

    for index, row in ports_df.iterrows():
        port_lat = row['Latitude']
        port_long = row['Longitude']
        closest_valid_lat_key = min(LAT_KEYS, key=lambda x: abs(x - port_lat))
        closest_valid_long_key = min(LONG_KEYS, key=lambda x: abs(x - port_long))

        for temprature_month_dfs, ports_month_temprature in zip(temprature_dfs, ports_temprature_per_month):
            port_temprature_value = get_temprature_or_chlorophyl_value(temprature_month_dfs, closest_valid_lat_key,
                                                                       closest_valid_long_key)
            ports_month_temprature.append(port_temprature_value)

        for chlorophyl_month_dfs, ports_month_chlorophyl in zip(chlorophyl_dfs, ports_chlorophyl_per_month):
            port_chlorophyl_value = get_temprature_or_chlorophyl_value(chlorophyl_month_dfs, closest_valid_lat_key,
                                                                       closest_valid_long_key)
            ports_month_chlorophyl.append(port_chlorophyl_value)

    for month_name, ports_month_temprature in zip(MONTHS_NAMES, ports_temprature_per_month):
        ports_df[f'NASA Temp {month_name}'] = ports_month_temprature

    for month_name, ports_month_chlorophyl in zip(MONTHS_NAMES, ports_chlorophyl_per_month):
        ports_df[f'NASA Chlorophyl {month_name}'] = ports_month_chlorophyl


def prepare_salinity_ceda_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    lat_long_pairs = df[['lat', 'lon']].to_numpy()
    coordinates_kd_tree = spatial.KDTree(lat_long_pairs)
    return df, lat_long_pairs, coordinates_kd_tree


# def prepare_salinity_woa_dataset(dataset_path):
#     df = pd.read_csv(dataset_path)
#     df.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'long'}, inplace=True)
#     df = df[['lat', 'long', '0', '5', '10', '15', '20']]
#     df['v'] = df[['0', '5', '10', '15', '20']].mean(axis=1)
#     df.to_csv(os.path.join(
#         OUTPUTS_DIR, os.path.basename(dataset_path)), index=False)
#     lat_long_pairs = df[['lat', 'long']].to_numpy()
#     coordinates_kd_tree = spatial.KDTree(lat_long_pairs)
#     return df, lat_long_pairs, coordinates_kd_tree


def get_salinity_value(dataset, lat_long_pair):
    lat, long = lat_long_pair[0], lat_long_pair[1]
    row_in_dataset = dataset.loc[(
        dataset['lon'] == long) & (dataset['lat'] == lat)]
    return row_in_dataset['sss'].item()


def add_ceda_salinity_to_ports_df(ports_df):
    salinity_datasets = [prepare_salinity_ceda_dataset(path) for path in SALINITY_CEDA_PATHS]
    ports_salinity_per_month = [[] for _ in range(12)]

    for index, row in ports_df.iterrows():
        port_coordinates = np.array((row['Latitude'], row['Longitude']))
        for salinity_dataset, ports_month_salinity in zip(salinity_datasets, ports_salinity_per_month):
            salinity_df, lat_long_pairs, coordinates_kd_tree = salinity_dataset
            closest_valid_lat_long_pair = lat_long_pairs[coordinates_kd_tree.query(port_coordinates)[1]]
            salinity_value = get_salinity_value(salinity_df, closest_valid_lat_long_pair)
            ports_month_salinity.append(salinity_value)

    for month_name, ports_month_salinity in zip(MONTHS_NAMES, ports_salinity_per_month):
        ports_df[f'CEDA Salinity {month_name}'] = ports_month_salinity


# def add_woa_salinity_to_ports_df(ports_df):
#     salinity_df, lat_long_pairs, coordinates_kd_tree = prepare_salinity_woa_dataset(
#         SALINITY_WOA_PATH)
#     salinity_values = []
#     for index, row in ports_df.iterrows():
#         port_coordinates = np.array((row['Latitude'], row['Longitude']))
#         closest_valid_lat_long_pair = lat_long_pairs[coordinates_kd_tree.query(
#             port_coordinates)[1]]
#         # if the closest point is not close enough
#         if np.linalg.norm(closest_valid_lat_long_pair - port_coordinates) > 1:
#             salinity_value = None
#         else:
#             salinity_value = get_salinity_value(
#                 salinity_df, closest_valid_lat_long_pair)
#         salinity_values.append(salinity_value)
#
#     ports_df['WOA Salinity'] = salinity_values


def add_conditions_from_2011_dataset(ports_df):
    ports_conditions_2011_df = pd.read_csv(PORTS_CONDITIONS_2011_PATH)
    ports_conditions_2011_df['PortName'] = ports_conditions_2011_df['PortName'].str.lower(
    )
    ports_conditions_2011_df = ports_conditions_2011_df.loc[ports_conditions_2011_df['PortName'].isin(
        ports_df['Port'])]
    ports_conditions_2011_df = ports_conditions_2011_df[[
        'PortName', 'MinTemp', 'MaxTemp', 'AnnualTemp', 'Salinity']]

    print(
        f"The ports 2011 conditions dataset contains {len(ports_conditions_2011_df)} relevant ports")

    ports_min_temp = []
    ports_max_temp = []
    ports_annual_temp = []
    ports_salinity = []

    for index, row in ports_df.iterrows():
        matching_port = ports_conditions_2011_df.loc[ports_conditions_2011_df['PortName'] == row['Port']]
        if matching_port.empty:
            ports_min_temp.append(None)
            ports_max_temp.append(None)
            ports_annual_temp.append(None)
            ports_salinity.append(None)
        else:
            ports_min_temp.append(matching_port['MinTemp'].item())
            ports_max_temp.append(matching_port['MaxTemp'].item())
            ports_annual_temp.append(matching_port['AnnualTemp'].item())
            ports_salinity.append(matching_port['Salinity'].item())

    ports_df['MinTemp_2011'] = ports_min_temp
    ports_df['MaxTemp_2011'] = ports_max_temp
    ports_df['AnnualTemp_2011'] = ports_annual_temp
    ports_df['Salinity_2011'] = ports_salinity


def main():
    final_routes_df = process_routes_dataset(SHIPS_ROUTES_DATASET_EXTENDED)

    ports_df = extract_ports_locations_df_from_routes_df(SHIPS_ROUTES_DATASET_EXTENDED)
    add_nasa_conditions_to_ports_df(ports_df)
    add_ceda_salinity_to_ports_df(ports_df)
    #add_woa_salinity_to_ports_df(ports_df)
    add_conditions_from_2011_dataset(ports_df)
    ports_df.to_csv(os.path.join(OUTPUTS_DIR, "combined_ports_conditions.csv"), index=False)


if __name__ == "__main__":
    main()
