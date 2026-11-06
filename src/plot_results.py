"""
plot_results.py
----------------
Draws a chart of vegetation health (NDVI) over time, compared with the
normal level for each calendar month, and colors each month by its
risk level. Saves the image to docs/ndvi_risk_chart.png so it can be shown
in the README.

Run this AFTER risk_score.py has produced data/risk_scores.csv.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # lets the script save images without a screen
import matplotlib.pyplot as plt

RISK_PATH = "data/risk_scores.csv"
OUT_PATH = "docs/ndvi_risk_chart.png"

# One color per risk level so the chart reads at a glance
COLORS = {"low": "#2e7d32", "moderate": "#f9a825", "high": "#c62828"}


def main():
    df = pd.read_csv(RISK_PATH, parse_dates=["date"]).sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Line: actual NDVI vs the average for that calendar month
    ax.plot(df["date"], df["ndvi_mean"], color="#555555", linewidth=1, label="NDVI (measured)")
    ax.plot(df["date"], df["ndvi_baseline"], color="#1565c0", linestyle="--",
            linewidth=1, label="NDVI (normal for that month)")

    # Dots: one per month, colored by risk level
    for level, color in COLORS.items():
        subset = df[df["risk_level"] == level]
        ax.scatter(subset["date"], subset["ndvi_mean"], color=color, s=28,
                   label=f"{level} risk", zorder=3)

    ax.set_title("Khorezm, Uzbekistan: vegetation health (NDVI) and drought-risk level")
    ax.set_xlabel("Month")
    ax.set_ylabel("NDVI (higher = greener)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved chart to {OUT_PATH}")


if __name__ == "__main__":
    main()
