import requests
import pandas as pd
from pathlib import Path

url = "https://api.open-meteo.com/v1/forecast?latitude=49.2827&longitude=-123.1207&current=temperature_2m,wind_speed_10m,precipitation&timezone=auto"

response = requests.get(url)

if response.status_code != 200:
    print("Request failed with status code:", response.status_code)
    exit()

data = response.json()
current = data['current']

temperature = current['temperature_2m']
wind_speed = current['wind_speed_10m']
precipitation = current.get('precipitation', 0)
timestamp = current["time"]

weather_df = pd.DataFrame([{
    'temperature': temperature,
    'wind_speed': wind_speed,
    'precipitation': precipitation,
    'timestamp': timestamp
}])

project_root = Path(__file__).resolve().parents[2]

csv_path = project_root / "data" / "weather.csv"

weather_df.to_csv(csv_path, index=False)

print("Weather data saved!")