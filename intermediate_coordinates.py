import searoute
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from utils import get_temperature_value, get_chlorophyll_value, get_salinity_value

MARITIME_MEASUREMENTS_TIME_INTERVAL = 24  # The time interval (in hours) to add measures between ports


def add_intermediate_coordinates(ship_winter_route):
    ship_winter_route.reset_index(drop=True, inplace=True)
    new_rows = []
    for index, row in ship_winter_route.iterrows():
        if index == 0 or row['Port'] == ship_winter_route.loc[index - 1]['Port']:
            continue

        origin = (ship_winter_route.loc[index - 1]['Longitude'], ship_winter_route.loc[index - 1]['Latitude'])
        destination = (row['Longitude'], row['Latitude'])

        origin_time = ship_winter_route.loc[index - 1]['Time']
        destination_time = row['Time']

        number_of_intermediate_coordinates = find_number_of_intermediate_days(origin_time, destination_time)
        if number_of_intermediate_coordinates == 0:
            continue

        intermediate_coordinates = find_coordinates_between_ports(origin, destination, number_of_intermediate_coordinates)
        intermediate_dates = get_intermediate_dates(origin_time, destination_time)
        for coordinate, date in zip(intermediate_coordinates, intermediate_dates):
            temperature = get_temperature_value(date.month, coordinate[::-1])
            chlorophyll = get_chlorophyll_value(date.month, coordinate[::-1])
            salinity = get_salinity_value(date.month, coordinate[::-1])
            new_row = pd.Series(data={'Ship': row['Ship'], 'Port': '-', 'Longitude': coordinate[0],
                                      'Latitude': coordinate[1], 'Time': date, 'Temperature': temperature,
                                      'Chlorophyll': chlorophyll, 'Salinity': salinity})
            new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows)
    extended_route = pd.concat([ship_winter_route, new_rows_df])
    extended_route.sort_values('Time', inplace=True, ignore_index=True)

    # Trim route to be exactly 30 days
    reference_time_value = ship_winter_route.iloc[0]['Time']
    for index, row in extended_route.iterrows():
        days_passed_from_first_row = (row['Time'].date() - reference_time_value.date()).days + 1
        if days_passed_from_first_row >= 30:
            sample_last_row_index = index
            break
    extended_route = extended_route.loc[extended_route.index <= sample_last_row_index]

    return extended_route


def find_number_of_intermediate_days(origin_time, destination_time):
    if origin_time.month == 1 and destination_time.month == 2:
        origin_day = -(31 - origin_time.day)
    else:
        origin_day = origin_time.day

    return (destination_time.day - origin_day) - 1


def find_coordinates_between_ports(origin, destination, number_of_intermediate_coordinates):
    """
    Calculate number_of_intermediate_coordinates evenly-spaced coordinates along the route returned by
    searoute package (which isn't evenly-spaced).
    Inspired by: https://mathematica.stackexchange.com/questions/223674/how-can-i-resample-a-list-of-x-y-data-for-evenly-spaced-points
    (converted to python with ChatGPT)
    """
    maritime_route = np.array(searoute.searoute(origin, destination, append_orig_dest=True)["geometry"]["coordinates"])
    lscpts = [line_scaled_coordinate(maritime_route, t) for t in np.linspace(0, 1, number_of_intermediate_coordinates + 2)]

    # Plotting the original points and the scaled coordinates
    # plt.plot(maritime_route[:, 0], maritime_route[:, 1], label='Original Points')
    # plt.scatter([p[0] for p in maritime_route], [p[1] for p in maritime_route], color='blue', label='Original Coordinates')
    # plt.scatter([p[0] for p in lscpts], [p[1] for p in lscpts], color='red', label='Scaled Coordinates')
    # plt.legend()
    # plt.show()

    return lscpts[1:-1]


def line_scaled_coordinate(points, t):
    return np.array([np.interp(t, np.linspace(0, 1, len(points)), points[:, i]) for i in range(points.shape[1])])


def get_intermediate_dates(origin_time, destination_time):
    delta = destination_time.date() - origin_time.date()

    dates = [pd.Timestamp(origin_time.date()) + timedelta(days=i) for i in range(1, delta.days)]
    return dates
