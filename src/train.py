
import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. KONFIGURASI PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "gold_features_latest.csv"
MODEL_DIR = ROOT / "models"
TRACKING_DB = ROOT / "mlflow.db"

EXPERIMENT_NAME = "gold-price-forecasting"
TARGET = "target_next_close_usd_per_gram"

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

CONFIGS = [
    {
        "name": "rf_baseline",
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_leaf": 2,
    },
    {
        "name": "rf_medium",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 2,
    },
    {
        "name": "rf_complex",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
    },
]


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 2. SIAPKAN DATA DAN TARGET HARI BERIKUTNYA
# ============================================================

def prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
            "Jalankan python src/preprocess.py terlebih dahulu."
        )

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Target hari berikutnya, bukan harga hari yang sama.
    df[TARGET] = df["Close_USD_per_gram"].shift(-1)

    # Baris terakhir tidak memiliki target hari berikutnya.
    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)

    if len(df) < 30:
        raise ValueError("Data terlalu sedikit untuk eksperimen ini.")

    # Pembagian kronologis: 80% awal untuk train, 20% akhir untuk test.
    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return df, X_train, X_test, y_train, y_test, split_idx


# ============================================================
# 3. JALANKAN TIGA EKSPERIMEN MLflow
# ============================================================

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Database tracking disimpan di root project.
    mlflow.set_tracking_uri(
        f"sqlite:///{TRACKING_DB.as_posix()}"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, X_train, X_test, y_train, y_test, split_idx = prepare_data()
    dataset_hash = sha256_file(DATA_PATH)

    print(f"Dataset       : {DATA_PATH.relative_to(ROOT)}")
    print(f"Total baris   : {len(df)}")
    print(f"Train         : {len(X_train)}")
    print(f"Test          : {len(X_test)}")
    print(f"Train sampai  : {df.iloc[split_idx - 1]['Date'].date()}")
    print(f"Test mulai    : {df.iloc[split_idx]['Date'].date()}")
    print(f"Dataset SHA256: {dataset_hash}")
    print()

    results = []

    for config in CONFIGS:
        run_name = config["name"]

        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=42,
            n_jobs=-1,
        )

        with mlflow.start_run(run_name=run_name) as run:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            rmse = float(
                np.sqrt(mean_squared_error(y_test, predictions))
            )
            mae = float(mean_absolute_error(y_test, predictions))
            r2 = float(r2_score(y_test, predictions))

            # Parameter eksperimen.
            mlflow.log_params(config)
            mlflow.log_param("target", TARGET)
            mlflow.log_param("split_strategy", "chronological_80_20")
            mlflow.log_param("random_state", 42)
            mlflow.log_param("feature_count", len(FEATURES))

            # Metrik evaluasi pada data test.
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)

            # Informasi dataset dan lineage.
            mlflow.log_param("dataset_path", str(DATA_PATH.relative_to(ROOT)))
            mlflow.log_param("dataset_sha256", dataset_hash)
            mlflow.log_param("dataset_rows", len(df))
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))

            mlflow.log_dict(
                {
                    "dataset_path": str(DATA_PATH.relative_to(ROOT)),
                    "dataset_sha256": dataset_hash,
                    "dataset_rows": len(df),
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "features": FEATURES,
                    "target": TARGET,
                    "train_end_date": str(
                        df.iloc[split_idx - 1]["Date"].date()
                    ),
                    "test_start_date": str(
                        df.iloc[split_idx]["Date"].date()
                    ),
                },
                "dataset_metadata.json",
            )

            # Simpan model sebagai artifact MLflow.
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=X_train.head(3),
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            results.append(
                {
                    "run_id": run.info.run_id,
                    "name": run_name,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "model": model,
                }
            )

            print(f"Eksperimen: {run_name}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE : {mae:.6f}")
            print(f"  R2  : {r2:.6f}")
            print(f"  Run : {run.info.run_id}")
            print()

    # Pilih model dengan RMSE test terendah.
    best = min(results, key=lambda item: item["rmse"])

    # Simpan salinan lokal model terbaik.
    import joblib

    model_path = MODEL_DIR / "best_gold_price_model.joblib"
    joblib.dump(best["model"], model_path)

    metadata = {
        "best_run_id": best["run_id"],
        "best_experiment": best["name"],
        "selection_metric": "test_rmse",
        "test_rmse": best["rmse"],
        "test_mae": best["mae"],
        "test_r2": best["r2"],
        "target": TARGET,
        "dataset_path": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": dataset_hash,
        "model_file": str(model_path.relative_to(ROOT)),
        "mlflow_experiment": EXPERIMENT_NAME,
    }

    with open(MODEL_DIR / "best_model_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("=" * 55)
    print("SELESAI: TIGA EKSPERIMEN TELAH DIJALANKAN")
    print(f"Model terbaik : {best['name']}")
    print(f"Best RMSE     : {best['rmse']:.6f}")
    print(f"Best MAE      : {best['mae']:.6f}")
    print(f"Best R2       : {best['r2']:.6f}")
    print(f"Model lokal   : {model_path.relative_to(ROOT)}")
    print(f"Metadata      : models/best_model_metadata.json")
    print("=" * 55)
    print("Untuk membuka MLflow UI, jalankan:")
    print("mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
