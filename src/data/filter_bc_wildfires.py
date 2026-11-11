import pandas as pd

df = pd.read_csv("data/viirs-jpss1_2025_Canada.csv")

# Keep only BC-ish coordinates
bc_df = df[
    (df["latitude"] >= 48)
    & (df["latitude"] <= 60)
    & (df["longitude"] >= -139)
    & (df["longitude"] <= -114)
]

# Convert date column
bc_df["acq_date"] = pd.to_datetime(
    bc_df["acq_date"]
)

# Fire season only
bc_df = bc_df[
    (bc_df["acq_date"] >= "2025-05-01")
    & (bc_df["acq_date"] <= "2025-09-30")
]

print("BC Fire Season Shape:")
print(bc_df.shape)

def assign_region(row):

    lat = row["latitude"]
    lon = row["longitude"]

    # Vancouver Island
    if lat < 50 and lon < -123:
        return "Vancouver Island"

    # Lower Mainland
    elif lat < 50:
        return "Lower Mainland"

    # Okanagan
    elif lon > -120:
        return "Okanagan"

    # Thompson
    elif lat < 51.5:
        return "Thompson"

    # Cariboo
    elif lat < 53:
        return "Cariboo"

    # Northern Interior
    elif lat < 55:
        return "Northern Interior"

    # Northeast
    elif lon > -123:
        return "Northeast"

    # Northwest
    else:
        return "Northwest"

bc_df["region"] = bc_df.apply(
    assign_region,
    axis=1
)

bc_df.to_csv(
    "data/bc_wildfires_2025.csv",
    index=False
)