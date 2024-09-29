import random
import os
import pandas as pd
import numpy as np

from utils import get_temperature_value, get_chlorophyll_value, get_salinity_value
from intermediate_coordinates import add_intermediate_coordinates

MIN_TEMPERATURE_IN_SAMPLED_ROUTES = 5
WINTER_MONTHS = [1, 2]
WINTER_MONTHS_TO_SUMMER_MONTH = {1: 7, 2: 8}


def sample_winter_subroutes(routes_df, routes_to_sample, outputs_dir, existing_route_ids=None):
    sampled_routes = []
    ships = list(set(routes_df['Ship']))

    sampled_routes_ids = [] if existing_route_ids is None else existing_route_ids
    while len(sampled_routes) < routes_to_sample:
        ship = random.choice(ships)
        chosen_year = random.choice([2019, 2020, 2021])

        if (ship, chosen_year) in sampled_routes_ids:
            continue

        ship_winter_route = routes_df.loc[(routes_df['Ship'] == ship) &
                                          (routes_df['Time'].dt.month.isin(WINTER_MONTHS)) &
                                          (routes_df['Time'].dt.year == chosen_year)].copy()

        if len(ship_winter_route) < 10:
            continue

        # Trim route to be 30 days
        sample_last_row_index = None
        reference_time_value = ship_winter_route.iloc[0]['Time']
        for index, row in ship_winter_route.iterrows():
            days_passed_from_first_row = (row['Time'].date() - reference_time_value.date()).days + 1
            if days_passed_from_first_row >= 30:
                sample_last_row_index = index
                break

        if sample_last_row_index is None:  # Winter route has data of less than 30 days
            continue
        ship_winter_route = ship_winter_route.loc[ship_winter_route.index <= sample_last_row_index]

        # If winter route has less than 10 data points, don't use it
        if len(ship_winter_route) < 10:
            continue

        extended_winter_route = add_intermediate_coordinates(ship_winter_route)

        if (extended_winter_route['Temperature'] < MIN_TEMPERATURE_IN_SAMPLED_ROUTES).any():
            continue

        # In case there are multiple data points in the same day, keep only 1.
        extended_winter_route['Date'] = extended_winter_route['Time'].dt.date
        extended_winter_route.drop_duplicates(subset=['Date'], inplace=True)
        extended_winter_route.drop(columns=['Date'], inplace=True)

        # Convert (absolute) time column to relative time (in hours)
        relative_times_in_hours = []
        for index, row in extended_winter_route.iterrows():
            relative_times_in_hours.append((row['Time'] - reference_time_value) / np.timedelta64(1, 'h'))
        extended_winter_route['Relative time (hours)'] = relative_times_in_hours

        extended_winter_route = extended_winter_route[['Ship', 'Port', 'Longitude', 'Latitude', 'Time',
                                                       'Relative time (hours)', 'Temperature', 'Chlorophyll', 'Salinity']]

        sampled_routes.append(extended_winter_route)
        sampled_routes_ids.append((ship, chosen_year))

    sampled_routes_df = pd.concat(sampled_routes, ignore_index=True)
    sampled_routes_df.rename(columns={'Temperature': 'Temperature (celsius)', 'Chlorophyll': 'Chlorophyll (mg/m^3)',
                                      'Salinity': 'Salinity (ppt)'}, inplace=True)

    sampled_routes_df.to_csv(os.path.join(outputs_dir, 'sampled_winter_routes.csv'), index=False, date_format="%d/%m/%y %H:%M:%S")

    print(f"Sampled {routes_to_sample} routes on January/February")
    return sampled_routes_df


def convert_to_summer_routes(routes_df, outputs_dir):
    times = []
    temperatures = []
    chlorophylls = []
    salinities = []

    for index, row in routes_df.iterrows():
        summer_month = WINTER_MONTHS_TO_SUMMER_MONTH[row['Time'].month]
        summer_time = row['Time'].replace(month=summer_month)
        coordinates = (row['Latitude'], row['Longitude'])
        summer_temperature = get_temperature_value(summer_month, coordinates)
        summer_chlorophyll = get_chlorophyll_value(summer_month, coordinates)

        if row['Port'] == '-':
            summer_salinity = get_salinity_value(summer_month, coordinates)
        else:  # In ports we have a fixed salinity all year
            summer_salinity = row['Salinity (ppt)']

        times.append(summer_time)
        temperatures.append(summer_temperature)
        chlorophylls.append(summer_chlorophyll)
        salinities.append(summer_salinity)

    routes_df['Time'] = times
    routes_df['Temperature (celsius)'] = temperatures
    routes_df['Chlorophyll (mg/m^3)'] = chlorophylls
    routes_df['Salinity (ppt)'] = salinities

    routes_df.to_csv(os.path.join(outputs_dir, 'sampled_summer_routes.csv'), index=False, date_format="%d/%m/%y %H:%M:%S")
    print(f"Converted winter routes to summer routes")
