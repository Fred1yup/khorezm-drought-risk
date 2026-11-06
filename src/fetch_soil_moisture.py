"""
fetch_soil_moisture.py
-----------------------
Pulls a monthly soil moisture time series for Khorezm province, Uzbekistan,
using NASA's AppEEARS service (SMAP satellite data), and saves it as
data/soil_moisture_khorezm.csv

Soil moisture is a more direct drought signal than rainfall alone --
rainfall can fall on already-saturated ground with little effect, while
low soil moisture means the ground itself is genuinely dry, which is what
actually stresses crops.

Requires the same free NASA Earthdata login as fetch_ndvi.py. Set it the
same way before running:

    import os
    os.environ["EARTHDATA_USERNAME"] = "your_username"
    os.environ["EARTHDATA_PASSWORD"] = "your_password"
"""

import os
import time
import requests
import pandas as pd
from datetime import date

APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

LATITUDE = 41.55
LONGITUDE = 60.63

START_DATE = "01-01-2017"
# SPL3SMP_E.005 only has data through Dec 2023 -- requesting beyond that
# causes NASA's server to reject the whole request. Capping it here so
# it stays within the dataset's actual coverage.
END_DATE = "12-01-2023"

# SPL3SMP_E = SMAP Enhanced L3 Soil Moisture, daily, 9km resolution
PRODUCT = "SPL3SMP_E.005"
LAYER = "Soil_Moisture_Retrieval_Data_AM_soil_moisture"


def login():
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment "
            "variables first (same login used in fetch_ndvi.py)."
        )
    response = requests.post(
        f"{APPEEARS_BASE}/login", auth=(username, password), timeout=30
    )
    response.raise_for_status()
    return response.json()["token"]


def submit_point_task(token):
    task = {
        "task_type": "point",
        "task_name": "khorezm_soil_moisture",
        "params": {
            "dates": [{"startDate": START_DATE, "endDate": END_DATE}],
            "layers": [{"product": PRODUCT, "layer": LAYER}],
            "coordinates": [
                {"latitude": LATITUDE, "longitude": LONGITUDE, "id": "khorezm_center"}
            ],
        },
    }
    response = requests.post(
        f"{APPEEARS_BASE}/task",
        json=task,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if not response.ok:
        print(f"NASA server said: {response.text}")
    response.raise_for_status()
    return response.json()["task_id"]


def wait_for_task(token, task_id):
    print("Waiting for NASA to process the request (usually a few minutes)...")
    while True:
        response = requests.get(
            f"{APPEEARS_BASE}/task/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
        status = response.json()["status"]
        print(f"  status: {status}")
        if status == "done":
            return
        if status in ("error", "expired"):
            raise RuntimeError(f"Task failed with status: {status}")
        time.sleep(20)


def download_results(token, task_id):
    response = requests.get(
        f"{APPEEARS_BASE}/bundle/{task_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    files = response.json()["files"]
    csv_file = next(f for f in files if f["file_name"].endswith("-results.csv"))
    file_id = csv_file["file_id"]

    file_response = requests.get(
        f"{APPEEARS_BASE}/bundle/{task_id}/{file_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    file_response.raise_for_status()

    raw_path = "data/soil_moisture_raw.csv"
    with open(raw_path, "wb") as f:
        f.write(file_response.content)
    return raw_path


def clean_and_aggregate(raw_path):
    df = pd.read_csv(raw_path)
    sm_col = [c for c in df.columns if "soil_moisture" in c.lower()][0]
    date_col = "Date"

    df = df[[date_col, sm_col]].rename(columns={date_col: "date", sm_col: "soil_moisture"})
    df["date"] = pd.to_datetime(df["date"])

    # Drop invalid/fill values (SMAP uses -9999 for missing data)
    df = df[df["soil_moisture"] > -1]

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["soil_moisture"].mean().reset_index()
    monthly = monthly.rename(columns={"month": "date"})
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m-01")
    return monthly


if __name__ == "__main__":
    print("Logging in to NASA Earthdata...")
    token = login()

    print("Submitting soil moisture request for Khorezm province...")
    task_id = submit_point_task(token)

    wait_for_task(token, task_id)

    print("Downloading results...")
    raw_path = download_results(token, task_id)

    print("Cleaning and aggregating to monthly values...")
    monthly_df = clean_and_aggregate(raw_path)

    out_path = "data/soil_moisture_khorezm.csv"
    monthly_df.to_csv(out_path, index=False)
    print(f"Saved {len(monthly_df)} rows to {out_path}")
