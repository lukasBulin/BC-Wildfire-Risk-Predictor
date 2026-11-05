import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FIRMS_API_KEY")

url = (
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    f"{API_KEY}/VIIRS_NOAA20_NRT/"
    f"-139,48,-114,60/5"
)

response = requests.get(url)

if response.status_code != 200:
    print("Error:", response.status_code)
    exit()

from io import StringIO

df = pd.read_csv(StringIO(response.text))

wildfire_df = df[[
    "acq_date",
    "latitude",
    "longitude",
    "confidence",
    "frp"
]]

wildfire_df.columns = [
    "date",
    "latitude",
    "longitude",
    "confidence",
    "frp"
]

os.makedirs("data", exist_ok=True)

wildfire_df.to_csv(
    "data/wildfires.csv",
    index=False
)

print("Wildfire data saved to data/wildfires.csv")