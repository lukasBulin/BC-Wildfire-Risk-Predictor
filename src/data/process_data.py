import pandas as pd

weather_df = pd.read_csv("data/weather.csv")
wildfire_df = pd.read_csv("data/wildfires.csv")

hotspots_per_day = (
    wildfire_df.groupby("date")
    .size()
    .reset_index(name="hotspot_count")
)

weather_daily = (
    weather_df.groupby("date")
    .agg({
        "temperature_max": "mean",
        "temperature_min": "mean",
        "precipitation": "mean"
    })
    .reset_index()
)

merged_df = pd.merge(
    weather_daily,
    hotspots_per_day,
    on="date",
    how="inner"
)

print("\nMerged Data:")
print(merged_df)

merged_df.to_csv(
    "data/processed_data.csv",
    index=False
)

print("Processed data saved to data/processed_data.csv")