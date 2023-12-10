from scipy import spatial
import pandas as pd

TEMPERATURE_NASA_PATHS = {
    1: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_01_formatted.CSV",
    2: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_02_formatted.CSV",
    3: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_03_formatted.CSV",
    4: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_04_formatted.CSV",
    5: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_05_formatted.CSV",
    6: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_06_formatted.CSV",
    7: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_07_formatted.CSV",
    8: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_08_formatted.CSV",
    9: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_09_formatted.CSV",
    10: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_10_formatted.CSV",
    11: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_11_formatted.CSV",
    12: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temperature data from NASA earth observations\2019\temprature_NASA_2019_12_formatted.CSV"
}

CHLOROPHYLL_NASA_PATHS = {
    1: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_01_formatted.CSV",
    2: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_02_formatted.CSV",
    3: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_03_formatted.CSV",
    4: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_04_formatted.CSV",
    5: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_05_formatted.CSV",
    6: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_06_formatted.CSV",
    7: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_07_formatted.CSV",
    8: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_08_formatted.CSV",
    9: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_09_formatted.CSV",
    10: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_10_formatted.CSV",
    11: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_11_formatted.CSV",
    12: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyll data from NASA earth observations\2019\chlorophyl_NASA_2019_12_formatted.CSV"
}

SALINITY_CEDA_PATHS = {
    1: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_01.csv",
    2: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_02.csv",
    3: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_03.csv",
    4: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_04.csv",
    5: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_05.csv",
    6: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_06.csv",
    7: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_07.csv",
    8: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_08.csv",
    9: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_09.csv",
    10: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_10.csv",
    11: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_11.csv",
    12: r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_12.csv"
}


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
