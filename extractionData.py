import os, json, pathlib
import cdsapi

cds_config = """
url: https://cds.climate.copernicus.eu/api
key: 06647836-535d-4ad4-bb8b-8c57449c226d
"""

path = pathlib.Path.home() / ".cdsapirc"
path.write_text(cds_config)

print("CDS API key configured")

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "specific_rain_water_content",
        "temperature"
    ],
    "year": [
        "2016", "2017", "2018",
        "2019", "2020"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "06:00", "12:00",
        "18:00"
    ],
    "pressure_level": ["1"],
    "data_format": "grib",
    "download_format": "zip"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
