import pandas as pd
import os

TEMPRATURE_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_01.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_02.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_03.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_04.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_05.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_06.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_07.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_08.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_09.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_10.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_11.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\temprature data from NASA earth observations\2019\temprature_NASA_2019_12.CSV"
]

CHLOROPHYL_PATHS = [
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_01.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_02.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_03.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_04.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_05.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_06.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_07.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_08.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_09.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_10.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_11.CSV",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\chlorophyl data from NASA earth observations\2019\chlorophyl_NASA_2019_12.CSV"
]

temprature_2022_january_df = pd.read_csv(TEMPRATURE_PATHS[0])
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
    for path in TEMPRATURE_PATHS:
        convert_dataset(path)

    for path in CHLOROPHYL_PATHS:
        convert_dataset(path)


if __name__ == "__main__":
    main()
