from scipy import spatial
import pandas as pd
import os

BASE_DIR = r"C:\Users\TalPNB22\OneDrive\Documents\University\Master\Ships"
ENVIRONMENT_CONDITIONS_BASE_DIR = os.path.join(BASE_DIR, "datasets from Doron", "environment conditions")

TEMPERATURE_FILE_PATTERN = os.path.join(ENVIRONMENT_CONDITIONS_BASE_DIR, r"temperature data from NASA earth observations", "2019", "temprature_NASA_2019_{:02d}_formatted.CSV")
TEMPERATURE_NASA_PATHS = {i: TEMPERATURE_FILE_PATTERN.format(i) for i in range(1,13)}

CHLOROPHYLL_FILE_PATTERN = os.path.join(ENVIRONMENT_CONDITIONS_BASE_DIR, r"chlorophyll data from NASA earth observations", "2019", "chlorophyl_NASA_2019_{:02d}_formatted.CSV")
CHLOROPHYLL_NASA_PATHS = {i: CHLOROPHYLL_FILE_PATTERN.format(i) for i in range(1,13)}

SALINITY_FILE_PATTERN = os.path.join(ENVIRONMENT_CONDITIONS_BASE_DIR, r"salinity data from CEDA", "2019 csv files", "salinity_2019_{:02d}.csv")
SALINITY_CEDA_PATHS = {i: SALINITY_FILE_PATTERN.format(i) for i in range(1,13)}


def read_condition_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    lat_lon_pairs = df[['lat', 'lon']].to_numpy()
    coordinates_kd_tree = spatial.KDTree(lat_lon_pairs)
    return df, coordinates_kd_tree


def get_closest_condition_value(condition_dataset, query_coordinates):
    condition_df, condition_coordinates_kd_tree = condition_dataset
    condition_closest_coordinate_index = condition_coordinates_kd_tree.query(query_coordinates)[1]
    condition_value = condition_df.loc[condition_closest_coordinate_index]['value']
    return condition_value


def get_temperature_value(month_number, query_coordinates):
    return get_closest_condition_value(TEMPERATURE_DATASETS[month_number], query_coordinates)


def get_chlorophyll_value(month_number, query_coordinates):
    return get_closest_condition_value(CHLOROPHYLL_DATASETS[month_number], query_coordinates)


def get_salinity_value(month_number, query_coordinates):
    return get_closest_condition_value(SALINITY_DATASETS[month_number], query_coordinates)


TEMPERATURE_DATASETS = {month_number: read_condition_dataset(path) for month_number, path in TEMPERATURE_NASA_PATHS.items()}
CHLOROPHYLL_DATASETS = {month_number: read_condition_dataset(path) for month_number, path in CHLOROPHYLL_NASA_PATHS.items()}
SALINITY_DATASETS = {month_number: read_condition_dataset(path) for month_number, path in SALINITY_CEDA_PATHS.items()}
