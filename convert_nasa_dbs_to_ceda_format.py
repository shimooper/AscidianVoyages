import pandas as pd
import os

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

temprature_2022_january_df = pd.read_csv(TEMPRATURE_JANUARY_PATHS[0])
LAT_KEYS = temprature_2022_january_df['lat/lon'].to_numpy(dtype=float).round(2)
LONG_KEYS = temprature_2022_january_df.columns[1:].to_numpy(dtype=float).round(2)


def convert_dataset(path):
    df = pd.read_csv(path)

    df.columns = ['lat'] + list(LONG_KEYS)
    df['lat'] = LAT_KEYS

    lats = []
    lons = []
    values = []
    for index, row in df.iterrows():
        for col in df.columns[1:]:
            value = row[col]
            if value != 99999:
                lats.append(row['lat'])
                lons.append(col)
                values.append(value)

    output_df = pd.DataFrame(data={'lat': lats, 'lon': lons, 'value': values})
    output_df.to_csv(f"{os.path.splitext(path)[0]}_formatted.csv", index=False)


def main():
    for month_paths in TEMPRATURE_PATHS:
        for path in month_paths:
            convert_dataset(path)

    for month_paths in CHLOROPHYL_PATHS:
        for path in month_paths:
            convert_dataset(path)


if __name__ == "__main__":
    main()
