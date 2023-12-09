import searoute
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MARITIME_MEASUREMENTS_TIME_INTERVAL = 24  # The time interval (in hours) to add measures between ports


def add_intermediate_coordinates(ship_winter_route):
    ship_winter_route.reset_index(drop=True, inplace=True)
    new_rows = []
    for index, row in ship_winter_route.iterrows():
        if index == 0:
            continue
        if row['Port'] != ship_winter_route.loc[index - 1]['Port']:
            origin = (ship_winter_route.loc[index - 1]['Longitude'], ship_winter_route.loc[index - 1]['Latitude'])
            destination = (row['Longitude'], row['Latitude'])

            number_of_intermediate_coordinates = find_number_of_intermediate_days(ship_winter_route.loc[index - 1]['time'], row['time'])
            intermediate_coordinates = find_coordinates_between_ports(origin, destination, number_of_intermediate_coordinates)
            pass

            # selected_coordinates_indexes = np.round(np.linspace(0, len(maritime_route) - 1, number_of_intermediate_coordinates)).astype(int)
            # selected_coordinates = maritime_route[selected_coordinates_indexes[1:-1]]  # Remove the origin and destination indexes (first and last)
            # for coordinate in selected_coordinates:
            #     new_row = pd.Series({'Ship': row['Ship'], 'Port': '-', 'Longitude': coordinate[0], 'Latitude': coordinate[1], })


def find_number_of_intermediate_days(origin_time, destination_time):
    origin_day = int(origin_time.strftime('%d'))
    destination_day = int(destination_time.strftime('%d'))
    if int(origin_time.strftime('%m')) == 1 and int(destination_time.strftime('%m')) == 2:
        origin_day = -(31 - origin_day)
    return (destination_day - origin_day) - 1


def find_coordinates_between_ports(origin, destination, number_of_intermediate_coordinates):
    # The origin and destination appear twice, so I remove the duplicates
    maritime_route = np.array(searoute.searoute(origin, destination, append_orig_dest=True)["geometry"]["coordinates"])
    lscpts = [line_scaled_coordinate(maritime_route, t) for t in np.linspace(0, 1, number_of_intermediate_coordinates + 2)]

    # Plotting the original points and the scaled coordinates
    plt.plot(maritime_route[:, 0], maritime_route[:, 1], label='Original Points')
    plt.scatter([p[0] for p in maritime_route], [p[1] for p in maritime_route], color='blue', label='Original Coordinates')
    plt.scatter([p[0] for p in lscpts], [p[1] for p in lscpts], color='red', label='Scaled Coordinates')
    plt.legend()
    plt.show()

    return lscpts[1:-1]


def line_scaled_coordinate(points, t):
    return np.array([np.interp(t, np.linspace(0, 1, len(points)), points[:, i]) for i in range(points.shape[1])])
