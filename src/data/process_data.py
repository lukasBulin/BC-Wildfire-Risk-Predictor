import pandas as pd

weather_df = pd.read_csv("data/weather.csv")
wildfire_df = pd.read_csv("data/bc_wildfires_2025.csv")

hotspots_per_day = (
    wildfire_df.groupby(
        ["acq_date", "region"]
    )
    .size()
    .reset_index(
        name="hotspot_count"
    )
)

frp_per_day = (
    wildfire_df.groupby(
        ["acq_date", "region"]
    )["frp"]
    .sum()
    .reset_index(
        name="total_frp"
    )
)

hotspots_per_day = hotspots_per_day.rename(
    columns={"acq_date": "date"}
)

frp_per_day = frp_per_day.rename(
    columns={"acq_date": "date"}
)


weather_daily = (
    weather_df.groupby(
        ["date", "region"]
    )
    .agg({
        "temperature_max":"mean",
        "temperature_min":"mean",
        "precipitation":"mean",
        "wind_speed_max":"mean"
    })
    .reset_index()
)

wildfire_daily = pd.merge(
    hotspots_per_day,
    frp_per_day,
    on=["date", "region"]
)

merged_df = pd.merge(
    weather_daily,
    wildfire_daily,
    on=["date", "region"],
    how="inner"
)

merged_df["date"] = pd.to_datetime(
    merged_df["date"]
)

merged_df["month"] = merged_df["date"].dt.month

merged_df["day_of_year"] = (
    merged_df["date"].dt.day_of_year
)

merged_df = merged_df.sort_values(
    ["region", "date"]
)

# Previous day hotspot count
merged_df["previous_day_hotspots"] = (
    merged_df.groupby("region")["hotspot_count"]
    .shift(1)
)

# Previous day fire radiative power
merged_df["previous_day_frp"] = (
    merged_df.groupby("region")["total_frp"]
    .shift(1)
)

# 3-day rolling average hotspots
merged_df["hotspots_3day_avg"] = (
    merged_df.groupby("region")["hotspot_count"]
    .transform(
        lambda x: (
            x.shift(1)
            .rolling(
                window=3,
                min_periods=1
            )
            .mean()
        )
    )
)

# 7-day rolling average hotspots
merged_df["hotspots_7day_avg"] = (
    merged_df.groupby("region")["hotspot_count"]
    .transform(
        lambda x: (
            x.shift(1)
            .rolling(
                window=7,
                min_periods=1
            )
            .mean()
        )
    )
)

merged_df = merged_df.dropna()

merged_df.to_csv(
    "data/processed_data.csv",
    index=False
)

print(merged_df.columns.tolist())
print(merged_df.shape)