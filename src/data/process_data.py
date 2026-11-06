import pandas as pd

weather_df = pd.read_csv("data/weather.csv")
wildfire_df = pd.read_csv("data/bc_wildfires_2025.csv")

hotspots_per_day = (
    wildfire_df.groupby("acq_date")
    .size()
    .reset_index(name="hotspot_count")
)

hotspots_per_day = hotspots_per_day.rename(
    columns={"acq_date": "date"}
)

weather_daily = (
    weather_df.groupby("date")
    .agg({
        "temperature_max": "mean",
        "temperature_min": "mean",
        "precipitation": "mean",
        "wind_speed_max": "mean"
    })
    .reset_index()
)

merged_df = pd.merge(
    weather_daily,
    hotspots_per_day,
    on="date",
    how="inner"
)

merged_df["date"] = pd.to_datetime(
    merged_df["date"]
)

merged_df["month"] = merged_df["date"].dt.month

merged_df["day_of_year"] = (
    merged_df["date"].dt.day_of_year
)

merged_df["previous_day_hotspots"] = (
    merged_df["hotspot_count"].shift(1)
)
merged_df = merged_df.dropna()

merged_df.to_csv(
    "data/processed_data.csv",
    index=False
)