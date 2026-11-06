"""
fetch_ndvi.py  (NASA version)
------------------------------
Pulls a monthly NDVI (vegetation health) time series for Khorezm province,
Uzbekistan, using NASA's AppEEARS service (MODIS satellite data), and
saves it as data/ndvi_khorezm.csv

NDVI (Normalized Difference Vegetation Index) ranges roughly -1 to 1.
Higher = healthier/greener vegetation. Falling NDVI over time, compared to
the historical average for that same month, is an early drought signal.

Requires: a free NASA Earthdata Login account (no card, no billing).
Sign up here: https://urs.earthdata.nasa.gov/users/new

Set your username and password as environment variables before running,
so they're not typed directly into the script:

    In Colab, run this in a cell BEFORE running this script:
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

# Center point of Khorezm province, Uzbekistan (near Urgench)
LATITUDE = 41.55
LONGITUDE = 60.63

START_DATE = "01-01-2017"  # AppEEARS wants MM-DD-YYYY
END_DATE = date.today().strftime("%m-%d-%Y")

# MOD13Q1 = MODIS Vegetation Indices, 16-day, 250m resolution
PRODUCT = "MOD13Q1.061"
LAYER = "_250m_16_days_NDVI"


def login():
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment "
            "variables first (see the docstring at the top of this file)."
        )
    response = requests.post(
        f"{APPEEARS_BASE}/login", auth=(username, password), timeout=30
    )
    response.raise_for_status()
    return response.json()["token"]


def submit_point_task(token):
    task = {
        "task_type": "point",
        "task_name": "khorezm_ndvi",
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

    # Find the CSV results file (there's usually one main results CSV)
    csv_file = next(f for f in files if f["file_name"].endswith("-results.csv"))
    file_id = csv_file["file_id"]

    file_response = requests.get(
        f"{APPEEARS_BASE}/bundle/{task_id}/{file_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    file_response.raise_for_status()

    raw_path = "data/ndvi_raw.csv"
    with open(raw_path, "wb") as f:
        f.write(file_response.content)
    return raw_path


def clean_and_aggregate(raw_path):
    df = pd.read_csv(raw_path)
    ndvi_col = [c for c in df.columns if "NDVI" in c][0]
    date_col = "Date"

    df = df[[date_col, ndvi_col]].rename(columns={date_col: "date", ndvi_col: "ndvi_mean"})
    df["date"] = pd.to_datetime(df["date"])

    # MODIS NDVI values are scaled by 10000 in the raw product -- convert
    # back to the standard -1 to 1 range.
    df["ndvi_mean"] = df["ndvi_mean"] / 10000.0

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["ndvi_mean"].mean().reset_index()
    monthly = monthly.rename(columns={"month": "date"})
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m-01")
    return monthly


if __name__ == "__main__":
    print("Logging in to NASA Earthdata...")
    token = login()

    print("Submitting NDVI request for Khorezm province...")
    task_id = submit_point_task(token)

    wait_for_task(token, task_id)

    print("Downloading results...")
    raw_path = download_results(token, task_id)

    print("Cleaning and aggregating to monthly values...")
    monthly_df = clean_and_aggregate(raw_path)

    out_path = "data/ndvi_khorezm.csv"
    monthly_df.to_csv(out_path, index=False)
    print(f"Saved {len(monthly_df)} rows to {out_path}")
