import os
import calendar
import pandas as pd
import numpy as np
from scipy import spatial

OUTPUTS_DIR = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\outputs\long_routes_dataset"

SHIPS_ROUTES_DATASET_EXTENDED = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\ships routes\ships routes (ports only) with locations of ports - extended.csv"
SHIPS_ROUTES_DATASET = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\ships routes\ships routes (ports only) with locations of ports.csv"

MONTHS_NAMES = calendar.month_name[1:]

TEMPERATURE_NASA_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_01_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_02_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_03_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_04_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_05_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_06_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_07_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_08_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_09_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_10_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_11_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_12_formatted.CSV"
]

CHLOROPHYLL_NASA_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_01_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_02_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_03_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_04_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_05_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_06_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_07_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_08_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_09_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_10_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_11_formatted.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_12_formatted.CSV"
]

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

PORTS_CONDITIONS_2011_PATH = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\Ports dataset with their temprature and salinity.csv"


def process_routes_dataset(routes_path):
    routes_df = pd.read_csv(routes_path)

    # Convert columns to right data types
    routes_df['Arrival (LT)'] = pd.to_datetime(routes_df['Arrival (LT)'], errors='coerce', dayfirst=True)
    routes_df['Departure (LT)'] = pd.to_datetime(routes_df['Departure (LT)'], errors='coerce', dayfirst=True)
    routes_df['Longitude'] = pd.to_numeric(routes_df['Longitude'], errors='coerce')
    routes_df['Latitude'] = pd.to_numeric(routes_df['Latitude'], errors='coerce')
    routes_df['Hours'] = pd.to_numeric(routes_df['Hours'], errors='coerce')

    mean_time_in_port_by_ship = routes_df.groupby('Ship').mean(numeric_only=True)['Hours']

    final_ships_routes_dfs = []
    for name, ship_df in routes_df.groupby('Ship'):
        # Impute missing Hours (time in port) values
        ship_df.loc[ship_df['Hours'].isna(), 'Hours'] = mean_time_in_port_by_ship[name]
        ship_df['Hours (time delta)'] = pd.to_timedelta(ship_df['Hours'].astype(str) + 'h', errors='coerce')

        # Impute missing Departure and Arrival values based on the imputed Hours values
        ship_df.loc[ship_df['Departure (LT)'].isna(), 'Departure (LT)'] = ship_df['Arrival (LT)'] + ship_df['Hours (time delta)']
        ship_df.loc[ship_df['Arrival (LT)'].isna(), 'Arrival (LT)'] = ship_df['Departure (LT)'] - ship_df['Hours (time delta)']

        # Prepare final route of the ship
        duplicated_ship_df = ship_df.copy()
        ship_df = ship_df[['Ship', 'Port', 'Arrival (LT)']]
        new_ship_df = ship_df.rename(columns={'Arrival (LT)': 'time'})
        duplicated_ship_df = duplicated_ship_df[['Ship', 'Port', 'Departure (LT)']]
        duplicated_ship_df.rename(columns={'Departure (LT)': 'time'}, inplace=True)
        final_ship_df = pd.concat([new_ship_df, duplicated_ship_df], ignore_index=True)
        final_ship_df.sort_values('time', ascending=True, inplace=True)

        # Convert (absolute) time column to relative time (in hours)
        reference_time_value = final_ship_df.iloc[0]['time']
        relative_times_in_hours = []
        for index, row in final_ship_df.iterrows():
            relative_times_in_hours.append((row['time'] - reference_time_value) / np.timedelta64(1, 'h'))
        final_ship_df['relative time'] = relative_times_in_hours
        final_ships_routes_dfs.append(final_ship_df)

    final_routes_df = pd.concat(final_ships_routes_dfs, ignore_index=True)
    final_routes_df.to_csv(os.path.join(OUTPUTS_DIR, "routes_processed.csv"), index=False, date_format="%d/%m/%y %H:%M:%S")
    print(f"There are {len(final_routes_df.groupby('Ship'))} ships in the routes dataset, "
          f"with a mean route length of {final_routes_df.groupby('Ship').count()['Port'].mean() / 2} ports")

    return final_routes_df


def extract_ports_locations_df_from_routes_df(routes_path):
    routes_df = pd.read_csv(routes_path)
    ports_in_routes_df = routes_df[['Port', 'Longitude', 'Latitude']]
    ports_in_routes_df = ports_in_routes_df.drop_duplicates()
    ports_in_routes_df['Port'] = ports_in_routes_df['Port'].str.lower()
    ports_in_routes_df.reset_index(drop=True, inplace=True)

    print(f'There are {len(ports_in_routes_df)} ports in the routes dataset')
    return ports_in_routes_df


def prepare_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    lat_lon_pairs = df[['lat', 'lon']].to_numpy()
    coordinates_kd_tree = spatial.KDTree(lat_lon_pairs)
    return df, lat_lon_pairs, coordinates_kd_tree


def get_value_by_lat_lon_pair(dataset, lat_lon_pair):
    lat, lon = lat_lon_pair[0], lat_lon_pair[1]
    row_in_dataset = dataset.loc[(dataset['lon'] == lon) & (dataset['lat'] == lat)]
    return row_in_dataset['value'].item()


def add_conditions_to_ports_df(ports_df):
    temperature_datasets = [prepare_dataset(path) for path in TEMPERATURE_NASA_PATHS]
    chlorophyll_datasets = [prepare_dataset(path) for path in CHLOROPHYLL_NASA_PATHS]
    salinity_datasets = [prepare_dataset(path) for path in SALINITY_CEDA_PATHS]

    ports_temperature_per_month = [[] for _ in range(12)]
    ports_chlorophyll_per_month = [[] for _ in range(12)]
    ports_salinity_per_month = [[] for _ in range(12)]

    for index, row in ports_df.iterrows():
        print(f"Extracting environment conditions for port {row['Port']} (#{index})")
        port_coordinates = np.array((row['Latitude'], row['Longitude']))
        for month_index in range(12):
            temperature_df, temperature_lat_long_pairs, temperature_coordinates_kd_tree = temperature_datasets[month_index]
            temperature_closest_valid_lat_long_pair = temperature_lat_long_pairs[temperature_coordinates_kd_tree.query(port_coordinates)[1]]
            temperature_value = get_value_by_lat_lon_pair(temperature_df, temperature_closest_valid_lat_long_pair)
            ports_temperature_per_month[month_index].append(temperature_value)

            chlorophyll_df, chlorophyll_lat_long_pairs, chlorophyll_coordinates_kd_tree = chlorophyll_datasets[month_index]
            chlorophyll_closest_valid_lat_long_pair = chlorophyll_lat_long_pairs[chlorophyll_coordinates_kd_tree.query(port_coordinates)[1]]
            chlorophyll_value = get_value_by_lat_lon_pair(chlorophyll_df, chlorophyll_closest_valid_lat_long_pair)
            ports_chlorophyll_per_month[month_index].append(chlorophyll_value)

            salinity_df, salinity_lat_long_pairs, salinity_coordinates_kd_tree = salinity_datasets[month_index]
            salinity_closest_valid_lat_long_pair = salinity_lat_long_pairs[salinity_coordinates_kd_tree.query(port_coordinates)[1]]
            salinity_value = get_value_by_lat_lon_pair(salinity_df, salinity_closest_valid_lat_long_pair)
            ports_salinity_per_month[month_index].append(salinity_value)

    for month_name, ports_month_temperature in zip(MONTHS_NAMES, ports_temperature_per_month):
        ports_df[f'NASA temperature {month_name}'] = ports_month_temperature

    for month_name, ports_month_chlorophyll in zip(MONTHS_NAMES, ports_chlorophyll_per_month):
        ports_df[f'NASA chlorophyll {month_name}'] = ports_month_chlorophyll

    for month_name, ports_month_salinity in zip(MONTHS_NAMES, ports_salinity_per_month):
        ports_df[f'CEDA salinity {month_name}'] = ports_month_salinity


def add_conditions_from_2011_dataset(ports_df):
    ports_conditions_2011_df = pd.read_csv(PORTS_CONDITIONS_2011_PATH)
    ports_conditions_2011_df['PortName'] = ports_conditions_2011_df['PortName'].str.lower()
    ports_conditions_2011_df = ports_conditions_2011_df.loc[ports_conditions_2011_df['PortName'].isin(ports_df['Port'])]
    ports_conditions_2011_df = ports_conditions_2011_df[['PortName', 'MinTemp', 'MaxTemp', 'AnnualTemp', 'Salinity']]

    print(f"The ports 2011 conditions dataset contains {len(ports_conditions_2011_df)} relevant ports")

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
    add_conditions_to_ports_df(ports_df)
    add_conditions_from_2011_dataset(ports_df)
    ports_df.to_csv(os.path.join(OUTPUTS_DIR, "combined_ports_conditions.csv"), index=False)


if __name__ == "__main__":
    main()
