import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load data
df = pd.read_csv("data/processed_data.csv")
wildfire_df = pd.read_csv("data/bc_wildfires_2025.csv")
importance_df = pd.read_csv("data/feature_importance.csv")

# Load trained model
model = joblib.load("data/wildfire_model.pkl")

st.title("BC Wildfire Risk Predictor")

st.caption(
    "Random Forest model trained on BC 2025 wildfire season data (R² = 0.71)"
)

# ======================================
# Prediction Section
# ======================================

st.header("Wildfire Prediction")

col1, col2 = st.columns(2)

with col1:
    temperature_max = st.number_input(
        "Maximum Temperature (°C)",
        value=25.0
    )

    temperature_min = st.number_input(
        "Minimum Temperature (°C)",
        value=15.0
    )

    precipitation = st.number_input(
        "Precipitation (mm)",
        value=0.0
    )

with col2:
    wind_speed_max = st.number_input(
        "Wind Speed (km/h)",
        value=20.0
    )

    month = st.slider(
        "Month",
        5,
        9,
        7
    )

    previous_day_hotspots = st.number_input(
        "Previous Day Hotspots",
        value=100
    )

day_of_year = st.slider(
    "Day Of Year",
    121,
    273,
    200
)

if st.button("Predict Wildfire Activity"):

    prediction = model.predict([[
        temperature_max,
        temperature_min,
        precipitation,
        wind_speed_max,
        month,
        day_of_year,
        previous_day_hotspots
    ]])

    predicted_hotspots = np.expm1(prediction[0])

    st.metric(
        "Predicted Hotspots",
        f"{predicted_hotspots:.0f}"
    )

    if predicted_hotspots < 100:
        st.success("Low Wildfire Risk")
    elif predicted_hotspots < 1000:
        st.warning("Moderate Wildfire Risk")
    else:
        st.error("High Wildfire Risk")

# ======================================
# Feature Importance
# ======================================

st.header("Model Insights")

st.subheader("Feature Importance")

st.bar_chart(
    importance_df.set_index("Feature")
)

# ======================================
# Dashboard Metrics
# ======================================

st.header("2025 Fire Season Overview")

col1, col2, col3, col4 = st.columns(4)

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

with col4:
    st.metric(
        "Average Wind Speed",
        round(df["wind_speed_max"].mean(), 1)
    )

# ======================================
# Charts
# ======================================

st.header("Hotspot Trends")

chart_df = df.set_index("date")

st.subheader("Daily Hotspot Count")

st.line_chart(
    chart_df["hotspot_count"]
)

st.subheader("Temperature vs Hotspot Count")

st.line_chart(
    chart_df[
        [
            "temperature_max",
            "hotspot_count"
        ]
    ]
)

st.subheader("Precipitation vs Hotspot Count")

st.line_chart(
    chart_df[
        [
            "precipitation",
            "hotspot_count"
        ]
    ]
)

# ======================================
# Map
# ======================================

st.header("British Columbia Wildfire Hotspots (2025)")

import pydeck as pdk

st.pydeck_chart(
    pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=54.5,
            longitude=-125.0,
            zoom=4.8,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=wildfire_df.sample(5000),
                get_position="[longitude, latitude]",
                get_radius=1000,
                get_fill_color=[255, 60, 60, 180],
                pickable=True,
            )
        ],
    )
)

# ======================================
# Data
# ======================================

st.header("Processed Dataset")

st.dataframe(df)

st.download_button(
    label="Download Processed Data",
    data=df.to_csv(index=False),
    file_name="processed_data.csv",
    mime="text/csv"
)