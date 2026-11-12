import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib

# ======================================
# LOAD DATA
# ======================================

model = joblib.load(
    "data/wildfire_model.pkl"
)

model_columns = joblib.load(
    "data/model_columns.pkl"
)

df = pd.read_csv("data/processed_data.csv")
importance_df = pd.read_csv("data/feature_importance.csv")

df["date"] = pd.to_datetime(df["date"])

region_locations = {
    "Lower Mainland": (49.2827, -123.1207),
    "Vancouver Island": (48.4284, -123.3656),
    "Okanagan": (49.8880, -119.4960),
    "Thompson": (50.6745, -120.3273),
    "Cariboo": (52.1415, -122.1417),
    "Northern Interior": (53.9171, -122.7497),
    "Northeast": (56.2468, -120.8477),
    "Northwest": (54.7804, -127.1743)
}

# ======================================
# TITLE
# ======================================

st.title("BC Wildfire Risk Predictor 🔥")

# ======================================
# TABS
# ======================================

tab1, tab2, tab3 = st.tabs([
    "Project Overview",
    "Wildfire Activity Explorer",
    "Regional Analysis"
])

# ======================================
# TAB 1
# ======================================

with tab1:

    st.header("Project Overview")
    st.info("""
    This project analyzes historical wildfire activity across British Columbia
    using Open-Meteo weather observations and NASA FIRMS wildfire detections.
    The province was divided into geographic regions and weather conditions
    were matched with regional wildfire hotspot activity.
    A Random Forest Regressor was trained using historical data from the
    2025 BC wildfire season.
    """)
    
    st.divider()
    st.header("Data Sources")
    st.info("""
    Historical weather observations were collected from
    eight representative locations across British Columbia
    using the Open-Meteo Archive API.
    These observations provide temperature, precipitation,
    and wind measurements used for wildfire forecasting.
    """)
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Weather Data")
  
        st.info("""
        **Source:** Open-Meteo

        **Regions**
        - Lower Mainland
        - Vancouver Island
        - Okanagan
        - Thompson
        - Cariboo
        - Northern Interior
        - Northeast
        - Northwest

        **Variables**
        - Maximum Temperature
        - Minimum Temperature
        - Precipitation
        - Wind Speed
        """)

    with col2:

        st.subheader("Wildfire Data")

        st.info("""
        **Source:** NASA FIRMS

        **Period**
        - May 1, 2025
        - September 30, 2025

        **Variables**
        - Latitude
        - Longitude
        - Detection Date
        - Fire Radiative Power (FRP)
        - Confidence
        """)

    st.divider()
    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Samples",
        len(df)
    )

    c2.metric(
        "Regions",
        len(df["region"].unique())
    )

    c3.metric(
        "Average Hotspots",
        round(df["hotspot_count"].mean(), 1)
    )

    c4.metric(
        "Peak Hotspots",
        int(df["hotspot_count"].max())
    )
    st.divider()
    st.header("Model Summary")

    c1, c2 = st.columns(2)

    c1.metric("R² Score", "0.71")
    c2.metric("MAE", "0.79")

    st.info("""

    This Random Forest Regressor predicts **regional wildfire hotspot activity**
    using weather conditions, wildfire intensity indicators, and recent wildfire trends.

    **Key Predictors**
    - Previous Day Fire Radiative Power (FRP)
    - Previous Day Hotspot Count
    - Recent Hotspot Trends (3-Day and 7-Day Averages)

    **Performance**
    - R² Score: **0.71**
    - MAE: **0.79**

    The model explains approximately **71% of the variation**
    in wildfire hotspot activity during the 2025 BC fire season.
    """)

    st.divider()
    st.header("Feature Importance")

    st.dataframe(
        importance_df,
        use_container_width=True
    )

# ======================================
# TAB 2
# ======================================

with tab2:

    st.header("Wildfire Activity vs Model Prediction")

    st.info("""
    Compare actual wildfire hotspot activity with the Random Forest model's
    predicted hotspot activity for each British Columbia region during the
    2025 fire season.

    Move the slider through the season to evaluate how closely the model
    matched real wildfire activity.
    """)

    # ======================================
    # DATE SELECTION
    # ======================================

    available_dates = sorted(
        df["date"].dt.date.unique()
    )

    selected_date = st.select_slider(
        "Select Date",
        options=available_dates
    )

    daily_df = df[
        df["date"].dt.date == selected_date
    ].copy()

    # ======================================
    # GENERATE PREDICTIONS
    # ======================================

    prediction_values = []

    for _, row in daily_df.iterrows():

        features = pd.DataFrame([row])

        features = pd.get_dummies(
            features,
            columns=["region"]
        )

        features = features.reindex(
            columns=model_columns,
            fill_value=0
        )

        prediction = model.predict(features)[0]

        predicted_hotspots = np.expm1(prediction)

        prediction_values.append(
            predicted_hotspots
        )

    daily_df["predicted_hotspots"] = prediction_values

    # ======================================
    # SUMMARY METRICS
    # ======================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Actual BC Hotspots",
            int(
                daily_df["hotspot_count"].sum()
            )
        )

    with col2:
        st.metric(
            "Predicted BC Hotspots",
            int(
                daily_df["predicted_hotspots"].sum()
            )
        )

    with col3:
        average_error = (
            abs(
                daily_df["hotspot_count"]
                -
                daily_df["predicted_hotspots"]
            )
        ).mean()

        st.metric(
            "Average Error",
            round(average_error, 1)
        )

    # ======================================
    # HELPERS
    # ======================================

    def create_map_df(
        dataset,
        hotspot_column
    ):

        rows = []

        for _, row in dataset.iterrows():

            lat, lon = region_locations[
                row["region"]
            ]

            hotspots = row[hotspot_column]

            if hotspots < 25:
                color = [0, 180, 0]

            elif hotspots < 100:
                color = [255, 180, 0]

            else:
                color = [220, 0, 0]

            rows.append({
                "region": row["region"],
                "lat": lat,
                "lon": lon,
                "hotspots": round(hotspots),
                "color": color,
                "radius": max(
                    np.sqrt(hotspots) * 5000,
                    10000
                )
            })

        return pd.DataFrame(rows)

    actual_map_df = create_map_df(
        daily_df,
        "hotspot_count"
    )

    predicted_map_df = create_map_df(
        daily_df,
        "predicted_hotspots"
    )

    # ======================================
    # MAPS
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Actual Activity")

        actual_layer = pdk.Layer(
            "ScatterplotLayer",
            data=actual_map_df,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True
        )

        actual_deck = pdk.Deck(
            layers=[actual_layer],
            initial_view_state=pdk.ViewState(
                latitude=54.5,
                longitude=-125,
                zoom=4.8,
                pitch=0,
                bearing=0
            ),
            tooltip={
                "html": """
                <b>Region:</b> {region}<br/>
                <b>Actual Hotspots:</b> {hotspots}
                """
            }
        )

        st.pydeck_chart(
            actual_deck,
            use_container_width=True
        )

    with col2:

        st.subheader("Model Prediction")

        prediction_layer = pdk.Layer(
            "ScatterplotLayer",
            data=predicted_map_df,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True
        )

        prediction_deck = pdk.Deck(
            layers=[prediction_layer],
            initial_view_state=pdk.ViewState(
                latitude=54.5,
                longitude=-125,
                zoom=4.8,
                pitch=0,
                bearing=0
            ),
            tooltip={
                "html": """
                <b>Region:</b> {region}<br/>
                <b>Predicted Hotspots:</b> {hotspots}
                """
            }
        )

        st.pydeck_chart(
            prediction_deck,
            use_container_width=True
        )

    # ======================================
    # LEGEND
    # ======================================

    st.markdown("""
    ### Risk Levels

    🟢 Low Activity: Less than 25 hotspots

    🟡 Moderate Activity: 25-99 hotspots

    🔴 High Activity: 100+ hotspots
    """)

    # ======================================
    # COMPARISON TABLE
    # ======================================

    st.subheader("Actual vs Predicted Activity")

    comparison_df = daily_df[
        [
            "region",
            "hotspot_count",
            "predicted_hotspots"
        ]
    ].copy()

    comparison_df["error"] = (
        comparison_df["hotspot_count"]
        -
        comparison_df["predicted_hotspots"]
    ).abs()

    comparison_df = comparison_df.rename(
        columns={
            "hotspot_count":
                "Actual Hotspots",
            "predicted_hotspots":
                "Predicted Hotspots",
            "error":
                "Absolute Error"
        }
    )

    st.dataframe(
        comparison_df.sort_values(
            "Actual Hotspots",
            ascending=False
        ),
        use_container_width=True
    )

# ======================================
# TAB 3
# ======================================

with tab3:

    st.header("Data Analysis")

    st.write("""
    Analyze wildfire activity for all of British Columbia or drill down
    into individual regions.
    """)

    # ======================================
    # BC WIDE ANALYSIS
    # ======================================

    st.subheader("British Columbia Overview")

    bc_df = (
        df.groupby("date")
        .agg({
            "hotspot_count": "sum",
            "temperature_max": "mean",
            "temperature_min": "mean",
            "precipitation": "mean",
            "wind_speed_max": "mean",
            "total_frp": "sum"
        })
        .reset_index()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average BC Hotspots",
            round(
                bc_df["hotspot_count"].mean(),
                1
            )
        )

    with col2:
        st.metric(
            "Peak BC Hotspots",
            int(
                bc_df["hotspot_count"].max()
            )
        )

    with col3:
        st.metric(
            "Average BC Temperature",
            round(
                bc_df["temperature_max"].mean(),
                1
            )
        )

    st.subheader("BC Wildfire Activity")

    st.line_chart(
        bc_df.set_index("date")[
            "hotspot_count"
        ]
    )

    st.subheader("BC Temperature vs Hotspots")

    st.line_chart(
        bc_df.set_index("date")[
            [
                "temperature_max",
                "hotspot_count"
            ]
        ]
    )

    st.subheader("BC Fire Radiative Power (FRP)")

    st.line_chart(
        bc_df.set_index("date")[
            "total_frp"
        ]
    )

    # ======================================
    # REGIONAL ANALYSIS
    # ======================================

    st.divider()

    st.subheader("Regional Analysis")

    selected_region = st.selectbox(
        "Select Region",
        sorted(df["region"].unique())
    )

    region_df = df[
        df["region"] == selected_region
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average Hotspots",
            round(
                region_df["hotspot_count"].mean(),
                1
            )
        )

    with col2:
        st.metric(
            "Maximum Hotspots",
            int(
                region_df["hotspot_count"].max()
            )
        )

    with col3:
        st.metric(
            "Average Temperature",
            round(
                region_df["temperature_max"].mean(),
                1
            )
        )

    with col4:
        st.metric(
            "Average Wind Speed",
            round(
                region_df["wind_speed_max"].mean(),
                1
            )
        )

    st.subheader(
        f"{selected_region} Hotspot Activity"
    )

    st.line_chart(
        region_df.set_index("date")[
            "hotspot_count"
        ]
    )

    st.subheader(
        f"{selected_region} Temperature"
    )

    st.line_chart(
        region_df.set_index("date")[
            "temperature_max"
        ]
    )

    st.subheader(
        f"{selected_region} Precipitation"
    )

    st.line_chart(
        region_df.set_index("date")[
            "precipitation"
        ]
    )

    # ======================================
    # RAW DATA
    # ======================================

    st.divider()

    st.subheader("Dataset")

    st.dataframe(
        region_df,
        use_container_width=True
    )

    st.download_button(
        label="Download Processed Dataset",
        data=df.to_csv(index=False),
        file_name="processed_data.csv",
        mime="text/csv"
    )