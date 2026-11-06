"""
fetch_weather.py
-----------------
Pulls monthly rainfall total and average temperature for Khorezm province,
Uzbekistan, from Open-Meteo's free historical weather API (no API key or
signup needed), and saves it as data/weather_khorezm.csv

Open-Meteo docs: https://open-meteo.com/en/docs/historical-weather-api
"""

import requests
import pandas as pd
from datetime import date

# Roughly the center of Khorezm province (near Urgench)
LATITUDE = 41.55
LONGITUDE = 60.63

START_DATE = "2017-01-01"
END_DATE = date.today().isoformat()

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_daily_weather():
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "precipitation_sum,temperature_2m_mean",
        "timezone": "auto",
    }
    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def to_monthly(daily_json):
    df = pd.DataFrame(
        {
            "date": daily_json["daily"]["time"],
            "precipitation_mm": daily_json["daily"]["precipitation_sum"],
            "temp_mean_c": daily_json["daily"]["temperature_2m_mean"],
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    monthly = (
        df.set_index("date")
        .resample("MS")  # month start
        .agg({"precipitation_mm": "sum", "temp_mean_c": "mean"})
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m-01")
    return monthly


if __name__ == "__main__":
    print("Fetching daily weather data for Khorezm province...")
    daily_json = fetch_daily_weather()

    print("Aggregating to monthly totals/averages...")
    monthly_df = to_monthly(daily_json)

    out_path = "data/weather_khorezm.csv"
    monthly_df.to_csv(out_path, index=False)
    print(f"Saved {len(monthly_df)} rows to {out_path}")
