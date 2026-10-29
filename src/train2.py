
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/processed/gold_features_latest.csv"
METADATA_PATH = ROOT / "models/best_model_metadata.json"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Ambil konfigurasi dan fitur dari metadata LK-06
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

df = pd.read_csv(DATA_PATH)
features = metadata["features"]
target = metadata["target"]

# Buat target harga emas pada hari perdagangan berikutnya
df[target] = df["Close_USD_per_gram"].shift(-1)

# Baris terakhir tidak memiliki target hari berikutnya,
# sehingga harus dikeluarkan dari data training/testing.
df = df.dropna(subset=features + [target]).copy()

X = df[features]
y = df[target]

# Pertahankan pembagian data kronologis seperti LK-06
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Parameter berbeda sebagai eksperimen pembaruan LK-07
params = {
    "n_estimators": 250,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "random_state": 42,
}

mlflow.set_experiment("gold-price-forecasting")

with mlflow.start_run(run_name="lk07_rf_trees_250") as run:
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_params(params)
    mlflow.log_metrics({
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
    })
    mlflow.log_param("dataset_path", "data/processed/gold_features_latest.csv")
    mlflow.log_param("target", target)
    mlflow.log_param("feature_count", len(features))

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=X_train.head(3),
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("Training LK-07 selesai.")
    print("Run ID:", run.info.run_id)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE : {mae:.4f}")
    print(f"Test R²  : {r2:.4f}")
