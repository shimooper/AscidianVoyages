import xarray as xr

NC_DATASETS = [
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20190115-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_01.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20190215-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_02.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20190315-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_03.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20190415-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_04.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20190515-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_05.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20190615-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_06.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20190715-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_07.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20190815-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_08.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20190915-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_09.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20191015-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_10.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20191115-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_11.csv"),
    (
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 nc files\ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20191215-fv2.31.nc",
    r"C:\Users\yairs\OneDrive\Documents\University\Master\Ships\datasets from Doron\environment conditions\salinity data from CEDA\2019 csv files\salinity_2019_12.csv")
]


def main():
    for nc_file, output_file in NC_DATASETS:
        DS = xr.open_dataset(nc_file)
        df = DS.to_dataframe()
        df = df.reset_index()
        filtered_df = df.loc[df['sss'].notnull()]
        filtered_df.rename(columns={'sss': 'value'}, inplace=True)
        filtered_df = filtered_df[['lat', 'lon', 'value']]
        filtered_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
