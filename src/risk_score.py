"""
risk_score.py
-------------
Step 1: rule-based drought/water-scarcity risk scoring.

No machine learning yet -- just clear, explainable logic. This is already
a legitimate working product: it compares this month's NDVI and rainfall
to the historical average for that same calendar month, and flags months
where both are running significantly below normal.

Run this AFTER fetch_ndvi.py and fetch_weather.py have produced:
  data/ndvi_khorezm.csv
  data/weather_khorezm.csv

Produces:
  data/risk_scores.csv
"""

import pandas as pd

NDVI_PATH = "data/ndvi_khorezm.csv"
WEATHER_PATH = "data/weather_khorezm.csv"
OUT_PATH = "data/risk_scores.csv"

# How far below the historical average counts as "concerning" (tune these
# once you see real data -- these are reasonable starting points).
NDVI_DROP_THRESHOLD = 0.10      # 10% below the historical average for that month
RAINFALL_DROP_THRESHOLD = 0.30  # 30% below the historical average for that month


def load_and_merge():
    ndvi = pd.read_csv(NDVI_PATH, parse_dates=["date"])
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["date"])
    df = pd.merge(ndvi, weather, on="date", how="inner")
    df["month"] = df["date"].dt.month
    return df


def add_historical_baselines(df):
    """For each calendar month (Jan, Feb, ...), compute the average NDVI
    and rainfall across all years -- this is the 'normal' to compare
    against."""
    baselines = df.groupby("month").agg(
        ndvi_baseline=("ndvi_mean", "mean"),
        rainfall_baseline=("precipitation_mm", "mean"),
    )
    return df.merge(baselines, on="month", how="left")


def score_risk(df):
    df["ndvi_pct_of_normal"] = df["ndvi_mean"] / df["ndvi_baseline"]
    df["rainfall_pct_of_normal"] = df["precipitation_mm"] / df["rainfall_baseline"]

    ndvi_low = df["ndvi_pct_of_normal"] < (1 - NDVI_DROP_THRESHOLD)
    rainfall_low = df["rainfall_pct_of_normal"] < (1 - RAINFALL_DROP_THRESHOLD)

    def classify(row_ndvi_low, row_rain_low):
        if row_ndvi_low and row_rain_low:
            return "high"
        elif row_ndvi_low or row_rain_low:
            return "moderate"
        else:
            return "low"

    df["risk_level"] = [
        classify(n, r) for n, r in zip(ndvi_low, rainfall_low)
    ]
    return df


if __name__ == "__main__":
    print("Loading NDVI and weather data...")
    df = load_and_merge()

    print("Computing historical monthly baselines...")
    df = add_historical_baselines(df)

    print("Scoring risk for each month...")
    df = score_risk(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print("\nRisk level counts:")
    print(df["risk_level"].value_counts())
    print("\nMost recent 6 months:")
    print(
        df[["date", "ndvi_pct_of_normal", "rainfall_pct_of_normal", "risk_level"]]
        .tail(6)
        .to_string(index=False)
    )
