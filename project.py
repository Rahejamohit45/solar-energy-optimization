import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ==================================================
# 1. LOAD DATASET
# ==================================================

print("=" * 60)
print("       SOLAR ENERGY OPTIMIZATION SYSTEM")
print("=" * 60)

df = pd.read_csv("solardata.csv")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

print("\n1. DATASET INFORMATION")

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nFirst 5 Records:")
print(df.head())


# ==================================================
# 2. DATA CLEANING
# ==================================================

print("\n2. DATA CLEANING")

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()

numeric_columns = [
    "temperature_c",
    "humidity_pct",
    "cloud_cover_pct",
    "solar_irradiance_w_m2",
    "solar_generation_kwh",
    "household_consumption_kwh",
    "solar_used_directly_kwh",
    "grid_import_kwh",
    "solar_export_kwh",
    "estimated_savings_inr"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

required_columns = [
    "timestamp",
    "temperature_c",
    "solar_irradiance_w_m2",
    "solar_generation_kwh",
    "household_consumption_kwh"
]

df = df.dropna(
    subset=required_columns
).copy()

df = df.sort_values("timestamp").reset_index(drop=True)

print("\nRows After Cleaning:", len(df))
print("Missing Values After Cleaning:", df.isnull().sum().sum())


# ==================================================
# 3. EXPLORATORY DATA ANALYSIS
# ==================================================

print("\n3. EXPLORATORY DATA ANALYSIS")

print("\nStatistical Summary:")
print(df[numeric_columns].describe().round(2))

print("\nAverage Solar Generation:",
      round(df["solar_generation_kwh"].mean(), 2),
      "kWh per hour")

print("Average Electricity Consumption:",
      round(df["household_consumption_kwh"].mean(), 2),
      "kWh per hour")

print("Maximum Solar Generation:",
      round(df["solar_generation_kwh"].max(), 2),
      "kWh")

print("Minimum Solar Generation:",
      round(df["solar_generation_kwh"].min(), 2),
      "kWh")


# ==================================================
# 4. CORRELATION ANALYSIS
# ==================================================

print("\n4. CORRELATION ANALYSIS")

correlation_columns = [
    "temperature_c",
    "humidity_pct",
    "cloud_cover_pct",
    "solar_irradiance_w_m2",
    "solar_generation_kwh",
    "household_consumption_kwh"
]

correlation = df[correlation_columns].corr()

print(correlation.round(2))

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Solar Energy Correlation Heatmap")
plt.tight_layout()
plt.show()


# ==================================================
# 5. MACHINE LEARNING
# PREDICT SOLAR GENERATION
# ==================================================

print("\n5. MACHINE LEARNING MODEL")

features = [
    "temperature_c",
    "humidity_pct",
    "cloud_cover_pct",
    "solar_irradiance_w_m2"
]

target = "solar_generation_kwh"

X = df[features]
y = df[target]

# Chronological split: use earlier records for training
# and later records for testing.
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nModel Evaluation")
print("Mean Absolute Error:", round(mae, 3), "kWh")
print("R2 Score:", round(r2, 3))


# ==================================================
# 6. PREDICT SOLAR GENERATION
# ==================================================

print("\n6. SOLAR GENERATION PREDICTION")

temperature = float(
    input("Enter temperature (C): ")
)

humidity = float(
    input("Enter humidity (%): ")
)

cloud_cover = float(
    input("Enter cloud cover (%): ")
)

irradiance = float(
    input("Enter solar irradiance (W/m2): ")
)

new_data = pd.DataFrame({
    "temperature_c": [temperature],
    "humidity_pct": [humidity],
    "cloud_cover_pct": [cloud_cover],
    "solar_irradiance_w_m2": [irradiance]
})

predicted_generation = model.predict(new_data)[0]

predicted_generation = max(
    0,
    predicted_generation
)

print(
    "\nPredicted Solar Generation:",
    round(predicted_generation, 3),
    "kWh for the hour"
)


# ==================================================
# 7. ELECTRICITY CONSUMPTION AND SAVINGS
# ==================================================

print("\n7. ELECTRICITY SAVINGS ANALYSIS")

# These are illustrative tariff assumptions.
grid_rate = 10.0
export_rate = 3.0

df["solar_used_directly_kwh"] = np.minimum(
    df["solar_generation_kwh"],
    df["household_consumption_kwh"]
)

df["grid_import_kwh"] = np.maximum(
    df["household_consumption_kwh"]
    - df["solar_generation_kwh"],
    0
)

df["solar_export_kwh"] = np.maximum(
    df["solar_generation_kwh"]
    - df["household_consumption_kwh"],
    0
)

df["bill_without_solar_inr"] = (
    df["household_consumption_kwh"]
    * grid_rate
)

df["grid_cost_inr"] = (
    df["grid_import_kwh"]
    * grid_rate
)

df["export_credit_inr"] = (
    df["solar_export_kwh"]
    * export_rate
)

df["bill_with_solar_inr"] = (
    df["grid_cost_inr"]
    - df["export_credit_inr"]
)

df["estimated_savings_inr"] = (
    df["bill_without_solar_inr"]
    - df["bill_with_solar_inr"]
)

print("\nTotal Solar Generation:",
      round(df["solar_generation_kwh"].sum(), 2),
      "kWh")

print("Total Electricity Consumption:",
      round(df["household_consumption_kwh"].sum(), 2),
      "kWh")

print("Total Solar Energy Used Directly:",
      round(df["solar_used_directly_kwh"].sum(), 2),
      "kWh")

print("Total Grid Electricity Imported:",
      round(df["grid_import_kwh"].sum(), 2),
      "kWh")

print("Total Solar Energy Exported:",
      round(df["solar_export_kwh"].sum(), 2),
      "kWh")

print("Total Estimated Savings: Rs.",
      round(df["estimated_savings_inr"].sum(), 2))


# ==================================================
# 8. DAILY AND MONTHLY ANALYSIS
# ==================================================

print("\n8. DAILY AND MONTHLY ANALYSIS")

df["date"] = df["timestamp"].dt.date

daily_report = df.groupby("date").agg({
    "solar_generation_kwh": "sum",
    "household_consumption_kwh": "sum",
    "grid_import_kwh": "sum",
    "solar_export_kwh": "sum",
    "estimated_savings_inr": "sum"
})

print("\nDaily Report:")
print(daily_report.head(5).round(2))

df["month"] = df["timestamp"].dt.to_period("M")

monthly_report = df.groupby("month").agg({
    "solar_generation_kwh": "sum",
    "household_consumption_kwh": "sum",
    "grid_import_kwh": "sum",
    "solar_export_kwh": "sum",
    "estimated_savings_inr": "sum"
})

print("\nMonthly Report:")
print(monthly_report.head(12).round(2))


# ==================================================
# 9. SOLAR INVESTMENT PAYBACK
# ==================================================

print("\n9. SOLAR INVESTMENT ANALYSIS")

installation_cost = float(
    input("Enter solar installation cost (Rs): ")
)

if installation_cost < 0:
    raise ValueError("Installation cost cannot be negative.")

# Annualize based on the number of observed days.
observed_days = (
    df["timestamp"].dt.normalize().nunique()
)

total_savings = df["estimated_savings_inr"].sum()

if observed_days > 0:
    estimated_yearly_savings = (
        total_savings / observed_days
    ) * 365
else:
    estimated_yearly_savings = 0

print("Estimated Yearly Savings: Rs.",
      round(estimated_yearly_savings, 2))

if estimated_yearly_savings > 0:

    payback_years = (
        installation_cost / estimated_yearly_savings
    )

    print("Estimated Payback Period:",
          round(payback_years, 2),
          "years")

else:
    print("Payback period cannot be calculated.")


# ==================================================
# 10. DATA VISUALIZATION
# ==================================================

print("\n10. GENERATING GRAPHS")

# Graph 1: Solar generation vs consumption

daily_report.plot(
    y=[
        "solar_generation_kwh",
        "household_consumption_kwh"
    ],
    figsize=(12, 5)
)

plt.title("Daily Solar Generation vs Consumption")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.tight_layout()
plt.show()


# Graph 2: Monthly savings

monthly_report["estimated_savings_inr"].plot(
    kind="bar",
    figsize=(12, 5)
)

plt.title("Monthly Estimated Electricity Savings")
plt.xlabel("Month")
plt.ylabel("Savings (Rs)")
plt.tight_layout()
plt.show()


# Graph 3: Temperature vs solar generation

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="temperature_c",
    y="solar_generation_kwh",
    alpha=0.4
)

plt.title("Temperature vs Solar Generation")
plt.xlabel("Temperature (C)")
plt.ylabel("Solar Generation (kWh)")
plt.tight_layout()
plt.show()


# Graph 4: Actual vs predicted solar generation

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4
)

plt.xlabel("Actual Solar Generation (kWh)")
plt.ylabel("Predicted Solar Generation (kWh)")
plt.title("Actual vs Predicted Solar Generation")

plt.tight_layout()
plt.show()


# ==================================================
# 11. SAVE RESULTS
# ==================================================

df.to_csv(
    "solar_analysis_results.csv",
    index=False
)

daily_report.to_csv(
    "daily_solar_report.csv"
)

monthly_report.to_csv(
    "monthly_solar_report.csv"
)

print("\nResults saved successfully:")
print("1. solar_analysis_results.csv")
print("2. daily_solar_report.csv")
print("3. monthly_solar_report.csv")

print("\n" + "=" * 60)
print("           PROJECT COMPLETED")
print("=" * 60)