import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/processed_data.csv")

df = pd.get_dummies(
    df,
    columns=["region"]
)

X = df.drop(
    columns=[
        "date",
        "hotspot_count",
        "total_frp"
    ]
)

import numpy as np

y = np.log1p(df["hotspot_count"])
print(df["hotspot_count"].describe())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "data/feature_importance.csv",
    index=False
)

print(
    feature_importance.sort_values(
        by="Importance",
        ascending=False
    )
)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print(results.head(20))

import joblib

joblib.dump(
    X.columns.tolist(),
    "data/model_columns.pkl"
)
