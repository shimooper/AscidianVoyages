import requests
import shutil
import os

urls_temprature = [
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822628&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_01.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822632&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_02.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822630&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_03.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822634&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_04.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822624&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_05.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822622&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_06.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822626&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_07.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822636&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_08.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822638&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_09.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822642&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_10.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822644&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_11.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822640&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "temprature_NASA_2019_12.csv"),
]

urls_chlorophyl = [
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822668&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_01.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822670&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_02.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822672&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_03.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822674&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_04.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822678&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_05.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822680&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_06.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822676&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_07.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822767&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_08.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822771&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_09.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822763&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_10.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822769&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_11.csv"),
    (r"https://neo.gsfc.nasa.gov/servlet/RenderData?si=1822765&cs=rgb&format=SS.CSV&width=3600&height=1800",
     "chlorophyl_NASA_2019_12.csv")
]

for url, local_file_name in urls_temprature + urls_chlorophyl:
    try:
        print(f'Try downloading {local_file_name}')
        with requests.get(url, stream=True) as r:
            with open(local_file_name, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    except:
        print(f'exception in downloading {local_file_name}')
        os.remove(local_file_name)
