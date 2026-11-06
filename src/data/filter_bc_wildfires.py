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

bc_df.to_csv(
    "data/bc_wildfires_2025.csv",
    index=False
)

print("Saved to data/bc_wildfires_2025.csv")