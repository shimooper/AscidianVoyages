import os
import calendar
import pandas as pd
import numpy as np
from dateutil import parser
import random
import json
import datetime
from scipy import spatial
from utils import read_environment_datasets, get_closest_condition_value

OUTPUTS_DIR = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\outputs_new"

SHIPS_ROUTES_DATASET = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\ships routes\ships routes (ports only) with locations of ports - extended.csv"

MONTHS_NAMES = calendar.month_name[1:]

PORTS_CONDITIONS_2011_PATH = r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\Ports dataset with their temprature and salinity (2011 paper).csv"

MIN_TEMPERATURE_IN_SAMPLED_ROUTES = 5
ROUTES_SAMPLES_COUNT = 20


def process_routes_dataset(routes_path):
    output_path = os.path.join(OUTPUTS_DIR, "routes_processed.csv")

    if os.path.exists(output_path):
        final_routes_df = pd.read_csv(output_path)
        final_routes_df['time'] = pd.to_datetime(final_routes_df['time'], dayfirst=True)
    else:
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
            arrivals_df = ship_df[['Ship', 'Port', 'Longitude', 'Latitude', 'Arrival (LT)']].rename(columns={'Arrival (LT)': 'time'})
            departures_df = ship_df[['Ship', 'Port', 'Longitude', 'Latitude', 'Departure (LT)']].rename(columns={'Departure (LT)': 'time'})
            final_ship_df = pd.concat([arrivals_df, departures_df], ignore_index=True)
            final_ship_df.sort_values('time', ascending=True, inplace=True)
            final_ship_df.reset_index(inplace=True, drop=True)
            final_ships_routes_dfs.append(final_ship_df)

        final_routes_df = pd.concat(final_ships_routes_dfs, ignore_index=True)
        final_routes_df.to_csv(output_path, index=False, date_format="%d/%m/%y %H:%M:%S")

    print(f"There are {len(final_routes_df.groupby('Ship'))} ships in the routes dataset, "
          f"with a mean route length of {final_routes_df.groupby('Ship').count()['Port'].mean() / 2} ports")
    return final_routes_df


def extract_ports_locations_df_from_routes_df(routes_path):
    routes_df = pd.read_csv(routes_path)
    ports_in_routes_df = routes_df[['Port', 'Longitude', 'Latitude']]
    ports_in_routes_df = ports_in_routes_df.drop_duplicates()
    ports_in_routes_df['Port'] = ports_in_routes_df['Port'].str.lower()
    ports_in_routes_df.set_index('Port', inplace=True)

    print(f'There are {len(ports_in_routes_df)} ports in the routes dataset')
    return ports_in_routes_df


def add_temperature_and_chlorophyll_to_ports_df(ports_df):
    temperature_datasets, chlorophyll_datasets, _ = read_environment_datasets()

    ports_temperature_per_month = [[] for _ in range(12)]
    ports_chlorophyll_per_month = [[] for _ in range(12)]

    for index, row in ports_df.iterrows():
        print(f"Extracting temperature and chlorophyll for port '{index}'")
        port_coordinates = np.array((row['Latitude'], row['Longitude']))
        for month_index in range(12):
            temperature_value = get_closest_condition_value(temperature_datasets[month_index], port_coordinates)
            ports_temperature_per_month[month_index].append(temperature_value)

            chlorophyll_value = get_closest_condition_value(chlorophyll_datasets[month_index], port_coordinates)
            ports_chlorophyll_per_month[month_index].append(chlorophyll_value)

    for month_name, ports_month_temperature in zip(MONTHS_NAMES, ports_temperature_per_month):
        ports_df[f'NASA temperature {month_name}'] = ports_month_temperature

    for month_name, ports_month_chlorophyll in zip(MONTHS_NAMES, ports_chlorophyll_per_month):
        ports_df[f'NASA chlorophyll {month_name}'] = ports_month_chlorophyll


def add_salinity_to_ports_df(ports_df):
    ports_conditions_2011_df = pd.read_csv(PORTS_CONDITIONS_2011_PATH)
    ports_conditions_2011_df['PortName'] = ports_conditions_2011_df['PortName'].str.lower()
    lat_lon_pairs = ports_conditions_2011_df[['Longitude', 'Latitude']].to_numpy()
    ports_2011_coordinates_kd_tree = spatial.KDTree(lat_lon_pairs)

    salinity = []
    for index, row in ports_df.iterrows():
        print(f"Extracting salinity for port '{index}'")
        port_coordinates = row['Longitude'], row['Latitude']
        closest_port_in_2011_dataset = ports_conditions_2011_df.loc[ports_2011_coordinates_kd_tree.query(port_coordinates)[1]]
        salinity.append(closest_port_in_2011_dataset['Salinity'])

    ports_df['Salinity'] = salinity


def find_statistics_on_ports_conditions(ports_df):
    temperature_columns = [f'NASA temperature {month}' for month in MONTHS_NAMES]
    chlorophyll_columns = [f'NASA chlorophyll {month}' for month in MONTHS_NAMES]

    temperature_values = ports_df[temperature_columns].to_numpy()
    chlorophyll_values = ports_df[chlorophyll_columns].to_numpy()
    salinity_values = ports_df['Salinity'].to_numpy()

    statistics = {
        'Temperature Min': temperature_values.min(),
        'Temperature Max': temperature_values.max(),
        'Chlorophyll Min': chlorophyll_values.min(),
        'Chlorophyll Max': chlorophyll_values.max(),
        'Salinity Min': salinity_values.min(),
        'Salinity Max': salinity_values.max()
    }

    with open(os.path.join(OUTPUTS_DIR, 'conditions_statistics.json'), 'w') as conditions_statistics_file:
        json.dump(statistics, conditions_statistics_file)


def add_conditions_to_routes_df(routes_df, ports_df):
    temperature_col = []
    chlorophyll_col = []
    salinity_col = []

    for index, row in routes_df.iterrows():
        port = row['Port'].lower()
        month = row['time'].strftime("%B")

        temperature = ports_df.loc[port, f'NASA temperature {month}']
        chlorophyll = ports_df.loc[port, f'NASA chlorophyll {month}']
        salinity = ports_df.loc[port, f'Salinity']

        temperature_col.append(temperature)
        chlorophyll_col.append(chlorophyll)
        salinity_col.append(salinity)

    routes_df['Temperature'] = temperature_col
    routes_df['Chlorophyll'] = chlorophyll_col
    routes_df['Salinity'] = salinity_col


def sample_subroutes(routes_df):
    sampled_routes = []
    ships = list(set(routes_df['Ship']))

    sampled_routes_ids = []
    while len(sampled_routes) < ROUTES_SAMPLES_COUNT:
        ship = random.choice(ships)
        ship_route = routes_df.loc[routes_df['Ship'] == ship]
        chosen_month = random.choice([1, 2])
        chosen_year = random.choice([2019, 2020, 2021])

        if (ship, chosen_year, chosen_month) in sampled_routes_ids:
            continue

        ship_route_month = ship_route.loc[(ship_route['time'].dt.month == chosen_month) & (ship_route['time'].dt.year == chosen_year)].copy()
        if len(ship_route_month) < 10:
            continue
        if (ship_route_month['Temperature'] < MIN_TEMPERATURE_IN_SAMPLED_ROUTES).any():
            continue
        if ship_route_month.iloc[-1]['time'] - ship_route_month.iloc[0]['time'] < datetime.timedelta(days=25):
            continue

        # Convert (absolute) time column to relative time (in hours)
        reference_time_value = ship_route_month.iloc[0]['time']
        relative_times_in_hours = []
        for index, row in ship_route_month.iterrows():
            relative_times_in_hours.append((row['time'] - reference_time_value) / np.timedelta64(1, 'h'))
        ship_route_month['relative time (hours)'] = relative_times_in_hours
        ship_route_month = ship_route_month[['Ship', 'Port', 'Longitude', 'Latitude', 'time', 'relative time (hours)',
                                             'Temperature', 'Chlorophyll', 'Salinity']]

        sampled_routes.append(ship_route_month)
        sampled_routes_ids.append((ship, chosen_year, chosen_month))

    sampled_routes_unified_df = pd.concat(sampled_routes, ignore_index=True)
    sampled_routes_unified_df.to_csv(os.path.join(OUTPUTS_DIR, 'sampled_routes.csv'), index=False, date_format="%d/%m/%y %H:%M:%S")


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    processed_routes_df = process_routes_dataset(SHIPS_ROUTES_DATASET)

    ports_conditions_path = os.path.join(OUTPUTS_DIR, "combined_ports_conditions.csv")
    if not os.path.exists(ports_conditions_path):
        ports_df = extract_ports_locations_df_from_routes_df(SHIPS_ROUTES_DATASET)
        add_temperature_and_chlorophyll_to_ports_df(ports_df)
        add_salinity_to_ports_df(ports_df)
        ports_df.to_csv(ports_conditions_path)
    else:
        ports_df = pd.read_csv(ports_conditions_path, index_col='Port')
    print(f"Extracted environment conditions for the {len(ports_df)} ports that were found in the routes dataset")

    find_statistics_on_ports_conditions(ports_df)

    add_conditions_to_routes_df(processed_routes_df, ports_df)
    processed_routes_df.to_csv(os.path.join(OUTPUTS_DIR, "routes_with_conditions.csv"), index=False)
    print("Added conditions to the routes")

    sample_subroutes(processed_routes_df)
    print(f"Sampled {ROUTES_SAMPLES_COUNT} routes on January/February")


if __name__ == "__main__":
    main()
