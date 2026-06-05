# download_era5_data.py

from pathlib import Path
import pathlib

############################################
# Configuration CDS API écrite automatiquement
############################################

cds_config = """
url: https://cds.climate.copernicus.eu/api
key: 06647836-535d-4ad4-bb8b-8c57449c226d
"""

path = pathlib.Path.home() / ".cdsapirc"
path.write_text(cds_config)

print(f"CDS API key configured in {path}")


############################################
# Configuration du téléchargement
############################################


class CFG:

    data_dir = "./data"

    variable = "temperature"
    pressure_level = "1000"

    train_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    val_years = [2021]
    test_years = [2022]

    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    times = ["00:00", "06:00", "12:00", "18:00"]

    # Europe
    area = [72, -25, 34, 40]

    # résolution ERA5
    grid = [0.25, 0.25]


############################################
# Téléchargement
############################################

DOWNLOAD = True

if DOWNLOAD:

    import cdsapi

    client = cdsapi.Client()

    out_dir = Path(CFG.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years_to_download = sorted(set(CFG.train_years + CFG.val_years + CFG.test_years))

    for year in years_to_download:

        for month in CFG.months:

            target = out_dir / f"era5_t1000_europe_{year}_{month}.nc"

            if target.exists():
                print(f"[skip] {target.name}")
                continue

            request = {
                "product_type": ["reanalysis"],
                "variable": [CFG.variable],
                "pressure_level": [CFG.pressure_level],
                "year": [str(year)],
                "month": [month],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": list(CFG.times),
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": list(CFG.area),
                "grid": list(CFG.grid),
            }

            print(f"Downloading {year}-{month} -> {target}")

            client.retrieve("reanalysis-era5-pressure-levels", request, str(target))

    print("Download finished")
