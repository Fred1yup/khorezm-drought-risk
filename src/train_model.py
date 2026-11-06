"""
train_model.py  (forecasting version)
---------------------------------------
This is the actual PREDICTION step -- it forecasts vegetation stress
2 months ahead, using only data you'd already have in hand right now
(rainfall, temperature, soil moisture), before the stress is visible in
the vegetation itself.

Why this is different from risk_score.py:
  risk_score.py compares THIS month's numbers to normal -- that's
  detection of a problem that has already started. This script instead
  learns the lag pattern: "when rainfall/soil moisture look like THIS,
  NDVI tends to drop like THAT about 2 months later" -- so it can
  forecast forward using only current/recent data.

Run this AFTER fetch_ndvi.py, fetch_weather.py, and
fetch_soil_moisture.py have all produced their CSV files in data/.

Produces:
  data/forecast_model.joblib   (the trained model, reusable)
  data/forecast_results.csv    (predictions vs actual, for review)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

NDVI_PATH = "data/ndvi_khorezm.csv"
WEATHER_PATH = "data/weather_khorezm.csv"
SOIL_MOISTURE_PATH = "data/soil_moisture_khorezm.csv"

FORECAST_HORIZON_MONTHS = 2   # how far ahead we're predicting NDVI
LAG_MONTHS = 3                 # how many past months of history to use as features

OUT_MODEL_PATH = "data/forecast_model.joblib"
OUT_RESULTS_PATH = "data/forecast_results.csv"


def load_and_merge():
    ndvi = pd.read_csv(NDVI_PATH, parse_dates=["date"])
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["date"])
    soil = pd.read_csv(SOIL_MOISTURE_PATH, parse_dates=["date"])

    df = ndvi.merge(weather, on="date", how="inner").merge(soil, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_lagged_features(df):
    """
    For each month, build features from the PAST few months (rainfall,
    temperature, soil moisture -- NOT NDVI, since NDVI is what we're
    trying to predict), and the TARGET is NDVI a few months in the
    FUTURE. This is what makes it a forecast instead of a same-month
    comparison.
    """
    feature_cols = ["precipitation_mm", "temp_mean_c", "soil_moisture"]
    rows = []

    for i in range(LAG_MONTHS, len(df) - FORECAST_HORIZON_MONTHS):
        features = {}
        for lag in range(1, LAG_MONTHS + 1):
            past_row = df.iloc[i - lag]
            for col in feature_cols:
                features[f"{col}_lag{lag}"] = past_row[col]

        target_row = df.iloc[i + FORECAST_HORIZON_MONTHS]
        features["target_ndvi"] = target_row["ndvi_mean"]
        features["target_date"] = target_row["date"]
        features["prediction_made_on"] = df.iloc[i]["date"]
        rows.append(features)

    return pd.DataFrame(rows)


def train_and_evaluate(feature_df):
    feature_cols = [c for c in feature_df.columns if c.startswith(
        ("precipitation_mm", "temp_mean_c", "soil_moisture")
    )]

    X = feature_df[feature_cols]
    y = feature_df["target_ndvi"]

    if len(feature_df) < 20:
        print(
            f"Only {len(feature_df)} data points available -- that's thin "
            "for machine learning. The model below will still run, but "
            "treat its output as a rough early signal, not a precise "
            "forecast, until you have more months of data."
        )

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, feature_df["target_date"], test_size=0.25, random_state=42
    )

    # Random forest: an ensemble of many small decision trees. Good fit
    # here because (a) it handles small/noisy data reasonably well, and
    # (b) we can use the SPREAD of predictions across its individual
    # trees as a simple confidence estimate -- important for being
    # honest about uncertainty rather than presenting one falsely-precise
    # number.
    model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = np.mean(np.abs(predictions - y_test))
    print(f"\nAverage forecast error on held-out test months: {mae:.3f} NDVI units")
    print("(NDVI ranges roughly 0 to 1 in this region -- compare this error to that scale)")

    return model, feature_cols


def predict_with_confidence(model, X_row):
    """Returns (predicted NDVI, confidence interval) using the spread of
    predictions across the forest's individual trees."""
    tree_predictions = np.array([tree.predict(X_row)[0] for tree in model.estimators_])
    mean_pred = tree_predictions.mean()
    std_pred = tree_predictions.std()
    return mean_pred, std_pred


def classify_risk(predicted_ndvi, historical_average_ndvi):
    ratio = predicted_ndvi / historical_average_ndvi
    if ratio < 0.85:
        return "high"
    elif ratio < 0.95:
        return "moderate"
    else:
        return "low"


if __name__ == "__main__":
    print("Loading and merging NDVI, weather, and soil moisture data...")
    df = load_and_merge()
    print(f"  {len(df)} months of merged data available")

    print(f"\nBuilding features (past {LAG_MONTHS} months -> NDVI {FORECAST_HORIZON_MONTHS} months ahead)...")
    feature_df = build_lagged_features(df)
    print(f"  {len(feature_df)} training examples built")

    print("\nTraining forecasting model...")
    model, feature_cols = train_and_evaluate(feature_df)

    import joblib
    joblib.dump({"model": model, "feature_cols": feature_cols, "lag_months": LAG_MONTHS}, OUT_MODEL_PATH)
    print(f"\nSaved trained model to {OUT_MODEL_PATH}")

    # Save a results file showing every prediction vs what actually happened
    feature_df.to_csv(OUT_RESULTS_PATH, index=False)
    print(f"Saved forecast details to {OUT_RESULTS_PATH}")

    # Demonstrate a live forecast using the MOST RECENT available months
    print("\n--- Example: forecasting forward from the most recent data ---")
    historical_avg_ndvi = df["ndvi_mean"].mean()
    latest_features = {}
    for lag in range(1, LAG_MONTHS + 1):
        row = df.iloc[-lag]
        for col in ["precipitation_mm", "temp_mean_c", "soil_moisture"]:
            latest_features[f"{col}_lag{lag}"] = row[col]
    X_latest = pd.DataFrame([latest_features])[feature_cols]

    pred_ndvi, uncertainty = predict_with_confidence(model, X_latest)
    risk = classify_risk(pred_ndvi, historical_avg_ndvi)
    print(f"Forecast for {FORECAST_HORIZON_MONTHS} months from now:")
    print(f"  Predicted NDVI: {pred_ndvi:.3f}  (+/- {uncertainty:.3f})")
    print(f"  Historical average for comparison: {historical_avg_ndvi:.3f}")
    print(f"  Risk level: {risk}")
