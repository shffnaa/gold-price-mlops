
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "gold_features_latest.csv"
MODEL_PATH = ROOT / "models" / "best_gold_price_model.joblib"
METADATA_PATH = ROOT / "models" / "best_model_metadata.json"

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close_USD_per_gram",
    "is_outlier_shock",
    "SMA_7",
    "SMA_30",
    "Rolling_Std_30",
    "pct_change",
    "RSI_14",
]

df = pd.read_csv(DATA_PATH)
df["target_next_day"] = df["Close_USD_per_gram"].shift(-1)
df = df.dropna(subset=FEATURES + ["target_next_day"]).copy()

with open(METADATA_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

model = joblib.load(MODEL_PATH)

split_index = int(len(df) * 0.8)
test = df.iloc[split_index:].copy()

X_test = test[FEATURES]
y_test = test["target_next_day"]
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("=== EVALUASI MODEL HARGA EMAS ===")
print("Model:", metadata.get("model_name", "Random Forest"))
print("Jumlah data uji:", len(test))
print(f"RMSE: {rmse:.6f} USD/gram")
print(f"MAE : {mae:.6f} USD/gram")
print(f"R²  : {r2:.6f}")

print("\n=== CONTOH PREDIKSI ===")
results = pd.DataFrame({
    "Tanggal fitur": test["Date"] if "Date" in test.columns else test.index,
    "Harga aktual besok (USD/gram)": y_test.values,
    "Prediksi (USD/gram)": predictions,
})
print(results.tail(5).to_string(index=False))
