import searoute

MARITIME_MEASUREMENTS_TIME_INTERVAL = 24  # The time interval (in hours) to add measures between ports


def add_intermediate_coordinates(final_ship_df):
    new_rows = []
    for index, row in final_ship_df.iterrows():
        if index == 0:
            continue
        if row['Port'] != final_ship_df.loc[index - 1]['Port']:
            origin = (final_ship_df.loc[index - 1]['Longitude'], final_ship_df.loc[index - 1]['Latitude'])
            destination = (row['Longitude'], row['Latitude'])
            time_diff = row['relative time (hours)'] - final_ship_df.loc[index - 1]['relative time (hours)']
            maritime_route = searoute.searoute(origin, destination, append_orig_dest=True)["geometry"]["coordinates"][1:-1]  # The origin and destination appear twice so I remove the duplicates
            number_of_intermediate_coordinates = time_diff / MARITIME_MEASUREMENTS_TIME_INTERVAL
            selected_coordinates_indexes = np.round(np.linspace(0, len(maritime_route) - 1, number_of_intermediate_coordinates)).astype(int)
            selected_coordinates = maritime_route[selected_coordinates_indexes[1:-1]]  # Remove the origin and destination indexes (first and last)
            for coordinate in selected_coordinates:
                new_row = pd.Series({'Ship': row['Ship'], 'Port': '-', 'Longitude': coordinate[0], 'Latitude': coordinate[1], })
