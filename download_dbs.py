import requests
import shutil
import os

urls_temprature_2022 = [(r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822113&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2022_02.csv"),
        (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822111&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2022_03.csv"),
        (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1826799&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2022_05.csv"),
        ("https://neo.gsfc.nasa.gov/servlet/RenderData?si=1826801&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2022_06.csv"),
        ("https://neo.gsfc.nasa.gov/servlet/RenderData?si=1844330&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2022_08.csv"),
        ("https://neo.gsfc.nasa.gov/servlet/RenderData?si=1845492&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2022_09.csv"),
        ("https://neo.gsfc.nasa.gov/servlet/RenderData?si=1846885&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2022_11.csv"),
        ("https://neo.gsfc.nasa.gov/servlet/RenderData?si=1848877&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2022_12.csv")]

urls_temprature = [
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822210&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2021_02.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822212&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2021_03.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822206&cs=rgb&format=SS.CSV&width=3600&height=1800","temprature_NASA_2021_05.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822204&cs=rgb&format=SS.CSV&width=3600&height=1800","temprature_NASA_2021_06.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822222&cs=rgb&format=SS.CSV&width=3600&height=1800","temprature_NASA_2021_08.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822220&cs=rgb&format=SS.CSV&width=3600&height=1800","temprature_NASA_2021_09.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822224&cs=rgb&format=SS.CSV&width=3600&height=1800","temprature_NASA_2021_11.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822214&cs=rgb&format=SS.CSV&width=3600&height=1800", "temprature_NASA_2021_12.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822414&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2020_02.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822412&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2020_03.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822408&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2020_05.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822418&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2020_06.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822410&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2020_08.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822424&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2020_09.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822422&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2020_11.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1826762&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2020_12.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822632&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_02.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822630&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_03.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822624&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_05.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822622&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_06.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822636&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_08.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822638&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_09.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822644&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_11.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822640&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_12.csv"),
]

urls_chlorophyl_2022 = [
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822053&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2022_02.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822055&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2022_03.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822130&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2022_05.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1845826&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2022_06.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1845814&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2022_08.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1845818&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2022_09.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1846838&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2022_11.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1849171&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2022_12.csv")
]

urls_chlorophyl = [
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822234&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2021_02.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822228&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2021_03.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822236&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2021_05.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822246&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2021_06.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822240&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2021_08.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822248&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2021_09.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822238&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2021_11.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822250&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2021_12.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822522&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2020_02.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822524&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2020_03.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822528&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2020_05.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822530&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2020_06.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822534&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2020_08.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822536&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2020_09.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822540&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2020_11.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822542&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2020_12.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822670&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2019_02.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822672&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2019_03.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822678&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2019_05.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822680&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2019_06.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822767&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2019_08.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822771&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2019_09.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822769&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2019_11.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822765&cs=rgb&format=SS.CSV&width=3600&height=1800", "chlorophyl_NASA_2019_12.csv")
]

for url, local_file_name in urls_chlorophyl:
    try:
        print(f'Try downloading {local_file_name}')
        with requests.get(url, stream=True) as r:
            with open(local_file_name, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    except:
        print(f'exception in downloading {local_file_name}')
        os.remove(local_file_name)
