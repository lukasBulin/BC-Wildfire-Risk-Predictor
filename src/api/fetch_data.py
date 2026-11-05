import requests
import pandas as pd
from pathlib import Path

cities = {
    "Vancouver": (49.2827, -123.1207),
    "Kamloops": (50.6745, -120.3273),
    "Kelowna": (49.8880, -119.4960),
    "Prince George": (53.9171, -122.7497)
}

all_weather_data = []

for city, (latitude, longitude) in cities.items():
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date=2026-11-01&end_date=2026-11-05&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=America/Vancouver"
    
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Request failed for {city} with status code:", response.status_code)
        exit()

    data = response.json()
    daily = data['daily']

    city_df = pd.DataFrame({
        "city": city,
        "date": daily["time"],
        "temperature_max": daily["temperature_2m_max"],
        "temperature_min": daily["temperature_2m_min"],
        "precipitation": daily["precipitation_sum"]
    })

    all_weather_data.append(city_df)

weather_df = pd.concat(all_weather_data, ignore_index=True)

project_root = Path(__file__).resolve().parents[2]

csv_path = project_root / "data" / "weather.csv"

weather_df.to_csv(csv_path, index=False)

print("Weather data saved!")