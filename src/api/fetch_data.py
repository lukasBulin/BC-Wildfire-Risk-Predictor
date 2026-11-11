import requests
import pandas as pd
from pathlib import Path

cities = [
    ("Vancouver", "Lower Mainland", 49.2827, -123.1207),
    ("Victoria", "Vancouver Island", 48.4284, -123.3656),
    ("Kelowna", "Okanagan", 49.8880, -119.4960),
    ("Kamloops", "Thompson", 50.6745, -120.3273),
    ("Williams Lake", "Cariboo", 52.1415, -122.1417),
    ("Prince George", "Northern Interior", 53.9171, -122.7497),
    ("Fort St John", "Northeast", 56.2468, -120.8477),
    ("Smithers", "Northwest", 54.7804, -127.1743)
]

all_weather_data = []

for city, region, latitude, longitude in cities:
    url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}"
    f"&longitude={longitude}"
    f"&start_date=2025-05-01"
    f"&end_date=2025-09-30"
    f"&daily=temperature_2m_max,"
    f"temperature_2m_min,"
    f"precipitation_sum,"
    f"wind_speed_10m_max"
    f"&timezone=America/Vancouver"
)

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Request failed for {city} with status code:", response.status_code)
        exit()

    data = response.json()
    daily = data['daily']

    city_df = pd.DataFrame({
        "city": city,
        "region": region,
        "date": daily["time"],
        "temperature_max": daily["temperature_2m_max"],
        "temperature_min": daily["temperature_2m_min"],
        "precipitation": daily["precipitation_sum"],
        "wind_speed_max": daily["wind_speed_10m_max"]
    })

    all_weather_data.append(city_df)

weather_df = pd.concat(all_weather_data, ignore_index=True)

project_root = Path(__file__).resolve().parents[2]

csv_path = project_root / "data" / "weather.csv"

weather_df.to_csv(csv_path, index=False)