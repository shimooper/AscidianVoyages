from scipy import spatial
import pandas as pd

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


def read_environment_datasets():
    temperature_datasets = [read_condition_dataset(path) for path in TEMPERATURE_NASA_PATHS]
    chlorophyll_datasets = [read_condition_dataset(path) for path in CHLOROPHYLL_NASA_PATHS]
    salinity_datasets = [read_condition_dataset(path) for path in SALINITY_CEDA_PATHS]

    return temperature_datasets, chlorophyll_datasets, salinity_datasets


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
