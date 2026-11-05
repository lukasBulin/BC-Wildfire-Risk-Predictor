import streamlit as st
import pandas as pd

df = pd.read_csv("data/processed_data.csv")
wildfire_df = pd.read_csv("data/wildfires.csv")

st.subheader("Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Hotspots",
        int(df["hotspot_count"].sum())
    )

with col2:
    st.metric(
        "Average Daily Hotspots",
        round(df["hotspot_count"].mean(), 1)
    )

with col3:
    st.metric(
        "Average Temperature",
        round(df["temperature_max"].mean(), 1)
    )

st.subheader("Daily Hotspot Count")

st.line_chart(
    df.set_index("date")["hotspot_count"]
)

st.subheader("Temperature vs Hotspot Count")

chart_df = df.set_index("date")

st.line_chart(
    chart_df[["temperature_max", "hotspot_count"]]
)

chart_df[
    ["temperature_max", "precipitation", "hotspot_count"]
]

st.subheader("Wildfire Hotspot Locations")

st.map(
    wildfire_df.rename(
        columns={
            "latitude": "lat",
            "longitude": "lon"
        }
    )
)

st.subheader("Processed Data")

st.dataframe(df)

st.download_button(
    label="Download Processed Data",
    data=df.to_csv(index=False),
    file_name="processed_data.csv",
    mime="text/csv"
)