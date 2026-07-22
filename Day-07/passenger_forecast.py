"""
Day 7 - Time Series Forecasting (Airline Passengers)
----------------------------------------------------------
First time series project. Every previous project treated each data point
as independent (a house's price doesn't depend on another house's price).
Here, order matters a LOT - this month's passenger count is directly
related to last month's, and there are yearly patterns (more travel in
summer, for example).

Dataset: the classic "Airline Passengers" dataset - real monthly totals of
international airline passengers from 1949 to 1960 (144 months). This is
one of the most famous datasets in time series analysis (Box & Jenkins,
1976) and comes bundled with seaborn.
Source: https://github.com/mwaskom/seaborn-data (flights.csv)

Concepts covered:
- Trend (long-term increase/decrease) vs seasonality (repeating yearly pattern)
- Train/test split for time series - MUST split by time, not randomly
  (you can't "randomly" pick test months, since predicting the past using
  the future would be cheating - the model needs to only ever see the past)
- Holt-Winters Exponential Smoothing (handles both trend AND seasonality)
- Comparing against a naive baseline (simple moving average)
"""

import os
import numpy as np
import pandas as pd
os.makedirs("images", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# 1. Load the real dataset
# ---------------------------
df = sns.load_dataset("flights")
df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str), format="%Y-%b")
df = df.sort_values("date").reset_index(drop=True)
passengers = df.set_index("date")["passengers"]

print(f"Dataset: {len(passengers)} months of real airline passenger data")
print(f"Date range: {passengers.index.min().strftime('%Y-%m')} to {passengers.index.max().strftime('%Y-%m')}")

# ---------------------------
# 2. Plot the raw data first - trend and seasonality are visible just by eye
# ---------------------------
plt.figure(figsize=(11, 5))
plt.plot(passengers.index, passengers.values)
plt.title("Monthly Airline Passengers (1949-1960) - Real Data")
plt.xlabel("Date")
plt.ylabel("Passengers (thousands)")
plt.tight_layout()
plt.savefig("images/raw_data.png")
print("Saved images/raw_data.png")

# ---------------------------
# 3. Train/test split BY TIME - last 12 months are the test set
# This is different from every previous project: you can't shuffle time
# series data, the model can only ever be trained on the past.
# ---------------------------
train = passengers[:-12]
test = passengers[-12:]

print(f"\nTraining on {len(train)} months, testing on the last {len(test)} months")

# ---------------------------
# 4. Baseline: naive forecast (just repeat the last known value)
# Always good to have a dumb baseline - if your fancy model can't beat this,
# it's not actually adding value.
# ---------------------------
naive_forecast = [train.iloc[-1]] * len(test)
naive_mae = mean_absolute_error(test, naive_forecast)

# ---------------------------
# 5. Holt-Winters Exponential Smoothing - handles trend AND seasonality
# ---------------------------
model = ExponentialSmoothing(
    train,
    trend="add",           # there's a clear upward trend in this data
    seasonal="add",        # there's a repeating yearly pattern
    seasonal_periods=12    # 12 months = 1 full seasonal cycle
)
fitted_model = model.fit()
hw_forecast = fitted_model.forecast(len(test))

hw_mae = mean_absolute_error(test, hw_forecast)
hw_rmse = np.sqrt(mean_squared_error(test, hw_forecast))

print(f"\nNaive baseline MAE: {round(naive_mae, 1)} passengers")
print(f"Holt-Winters MAE:   {round(hw_mae, 1)} passengers")
print(f"Holt-Winters RMSE:  {round(hw_rmse, 1)} passengers")
improvement = round((1 - hw_mae / naive_mae) * 100, 1)
print(f"Holt-Winters beats the naive baseline by {improvement}%")

# ---------------------------
# 6. Plot actual vs predicted for the test period
# ---------------------------
plt.figure(figsize=(11, 5))
plt.plot(train.index, train.values, label="Training data")
plt.plot(test.index, test.values, label="Actual (test)", marker="o")
plt.plot(test.index, hw_forecast, label="Holt-Winters forecast", marker="x")
plt.plot(test.index, naive_forecast, label="Naive baseline", linestyle="--")
plt.title("Forecast vs Actual - Last 12 Months")
plt.xlabel("Date")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.tight_layout()
plt.savefig("images/forecast_vs_actual.png")
print("Saved images/forecast_vs_actual.png")

# ---------------------------
# 7. Forecast 12 months into the FUTURE (beyond the dataset entirely)
# Refit on the FULL dataset first so the forecast uses all available real data
# ---------------------------
full_model = ExponentialSmoothing(
    passengers, trend="add", seasonal="add", seasonal_periods=12
).fit()
future_forecast = full_model.forecast(12)

print("\n=== Forecast for the next 12 months (beyond the dataset) ===")
for date, value in zip(future_forecast.index, future_forecast.values):
    print(f"{date.strftime('%Y-%m')}: {round(value)} passengers")
