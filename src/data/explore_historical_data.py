import pandas as pd

df = pd.read_csv("data/viirs-jpss1_2025_Canada.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())