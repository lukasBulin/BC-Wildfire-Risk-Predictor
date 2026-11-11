import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# ======================================
# LOAD DATA
# ======================================

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

st.title("🔥 BC Wildfire Risk Predictor")

# ======================================
# TABS
# ======================================

tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Wildfire Map",
    "Data Analysis"
])

# ======================================
# TAB 1
# ======================================

with tab1:

    st.header("Project Overview")

    st.markdown("""
    This project analyzes historical wildfire activity across British Columbia
    using Open-Meteo weather observations and NASA FIRMS wildfire detections.

    The province was divided into geographic regions and weather conditions
    were matched with regional wildfire hotspot activity.

    A Random Forest Regressor was trained using historical data from the
    2025 BC wildfire season.
    """)

    st.header("Data Sources")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Weather Data")

        st.markdown("""
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

        st.markdown("""
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

    st.header("Model Performance")

    c1, c2 = st.columns(2)

    c1.metric("R² Score", "0.71")
    c2.metric("MAE", "0.79")

    st.info(
        "The model explains approximately 71% of the variation in wildfire "
        "hotspot activity during the 2025 BC fire season."
    )

    st.header("Feature Importance")

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

# ======================================
# TAB 2
# ======================================

with tab2:

    st.header("BC Wildfire Activity Explorer")

    st.write("""
    Explore wildfire hotspot activity across British Columbia during the
    2025 fire season.

    Move through the season using the slider below. The map displays
    hotspot activity by region, where circle size and color represent
    wildfire activity levels.
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
    ]

    # ======================================
    # DAILY METRICS
    # ======================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "BC Hotspots",
            int(daily_df["hotspot_count"].sum())
        )

    with col2:
        st.metric(
            "Average Temperature",
            round(
                daily_df["temperature_max"].mean(),
                1
            )
        )

    with col3:

        busiest_region = daily_df.loc[
            daily_df["hotspot_count"].idxmax(),
            "region"
        ]

        st.metric(
            "Most Active Region",
            busiest_region
        )

    # ======================================
    # BUILD MAP DATA
    # ======================================

    map_rows = []

    for _, row in daily_df.iterrows():

        lat, lon = region_locations[row["region"]]

        hotspot_count = row["hotspot_count"]

        # Risk color
        if hotspot_count < 25:
            color = [0, 200, 0]          # Green

        elif hotspot_count < 100:
            color = [255, 200, 0]        # Yellow

        else:
            color = [255, 0, 0]          # Red

        map_rows.append({
            "region": row["region"],
            "lat": lat,
            "lon": lon,
            "hotspot_count": hotspot_count,
            "color": color,
            "radius": (
                np.sqrt(hotspot_count) * 5000
                )
        })

    map_df = pd.DataFrame(map_rows)

    # ======================================
    # MAP
    # ======================================

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.8
    )

    view_state = pdk.ViewState(
    latitude=54.5,
    longitude=-125.0,
    zoom=4.8,
    pitch=0,
    bearing=0
)

    deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "html": """
        <b>Region:</b> {region}<br/>
        <b>Hotspots:</b> {hotspot_count}
        """
    }
)

    st.pydeck_chart(
        deck,
        use_container_width=True
    )

    # ======================================
    # LEGEND
    # ======================================

    st.markdown("""
    **Risk Levels**

    🟢 Low Activity: Less than 25 hotspots

    🟡 Moderate Activity: 25 to 99 hotspots

    🔴 High Activity: 100+ hotspots
    """)

    # ======================================
    # REGIONAL BREAKDOWN
    # ======================================

    st.subheader("Regional Activity")

    st.dataframe(
        daily_df[
            [
                "region",
                "hotspot_count",
                "temperature_max",
                "precipitation",
                "wind_speed_max"
            ]
        ]
        .sort_values(
            "hotspot_count",
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