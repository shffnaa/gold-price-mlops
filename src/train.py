import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "gold_features_latest.csv"
MODEL_DIR = ROOT / "models"
TRACKING_DB = ROOT / "mlflow.db"

EXPERIMENT_NAME = "gold-price-forecasting"
TARGET = "target_next_close_usd_per_gram"

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Close_USD_per_gram", "is_outlier_shock",
    "SMA_7", "SMA_30", "Rolling_Std_30",
    "pct_change", "RSI_14"
]

CONFIGS = [
    {"name": "rf_baseline", "n_estimators": 50, "max_depth": 5,
     "min_samples_leaf": 2, "min_samples_split": 2},
    {"name": "rf_medium", "n_estimators": 100, "max_depth": 10,
     "min_samples_leaf": 2, "min_samples_split": 2},
    {"name": "rf_complex", "n_estimators": 200, "max_depth": None,
     "min_samples_leaf": 1, "min_samples_split": 2},
    {"name": "rf_trees_150", "n_estimators": 150, "max_depth": 10,
     "min_samples_leaf": 2, "min_samples_split": 2},
    {"name": "rf_trees_300", "n_estimators": 300, "max_depth": 10,
     "min_samples_leaf": 2, "min_samples_split": 2},
    {"name": "rf_depth_5_leaf_1", "n_estimators": 100, "max_depth": 5,
     "min_samples_leaf": 1, "min_samples_split": 2},
    {"name": "rf_depth_15", "n_estimators": 100, "max_depth": 15,
     "min_samples_leaf": 2, "min_samples_split": 2},
    {"name": "rf_unlimited_leaf_2", "n_estimators": 100, "max_depth": None,
     "min_samples_leaf": 2, "min_samples_split": 2},
    {"name": "rf_split_5", "n_estimators": 100, "max_depth": 10,
     "min_samples_leaf": 2, "min_samples_split": 5},
    {"name": "rf_leaf_4", "n_estimators": 100, "max_depth": 10,
     "min_samples_leaf": 4, "min_samples_split": 2},
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================
# 2. PERSIAPAN DATA
# ============================================================

def prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df[TARGET] = df["Close_USD_per_gram"].shift(-1)
    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)

    if len(df) < 30:
        raise ValueError("Data terlalu sedikit untuk eksperimen ini.")

    # Chronological split: 64% train, 16% validation, 20% test.
    train_end = int(len(df) * 0.64)
    validation_end = int(len(df) * 0.80)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_validation = X.iloc[train_end:validation_end]
    y_validation = y.iloc[train_end:validation_end]

    # Test set sengaja tidak digunakan dalam proses training.
    X_train_validation = X.iloc[:validation_end]
    y_train_validation = y.iloc[:validation_end]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    return (
        df,
        X_train, y_train,
        X_validation, y_validation,
        X_train_validation, y_train_validation,
        X_test, y_test,
        train_end, validation_end
    )


# ============================================================
# 3. TRAINING DAN PEMILIHAN MODEL
# ============================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    (
        df,
        X_train, y_train,
        X_validation, y_validation,
        X_train_validation, y_train_validation,
        X_test, y_test,
        train_end, validation_end
    ) = prepare_data()

    dataset_hash = sha256_file(DATA_PATH)

    print("=== CHRONOLOGICAL DATA SPLIT ===")
    print(f"Total data       : {len(df)}")
    print(f"Training         : {len(X_train)}")
    print(f"Validation      : {len(X_validation)}")
    print(f"Test             : {len(X_test)}")
    print(f"Train sampai     : {df.iloc[train_end - 1]['Date'].date()}")
    print(f"Validation mulai : {df.iloc[train_end]['Date'].date()}")
    print(f"Test mulai       : {df.iloc[validation_end]['Date'].date()}")
    print(f"Dataset SHA256   : {dataset_hash}")
    print()

    results = []

    for index, config in enumerate(CONFIGS, start=1):
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            min_samples_split=config["min_samples_split"],
            random_state=42,
            n_jobs=-1
        )

        with mlflow.start_run(run_name=config["name"]) as run:
            model.fit(X_train, y_train)
            predictions = model.predict(X_validation)

            rmse = float(np.sqrt(mean_squared_error(
                y_validation, predictions
            )))
            mae = float(mean_absolute_error(y_validation, predictions))
            r2 = float(r2_score(y_validation, predictions))

            mlflow.log_params(config)
            mlflow.log_param("target", TARGET)
            mlflow.log_param("split_strategy", "chronological_64_16_20")
            mlflow.log_param("random_state", 42)
            mlflow.log_param("feature_count", len(FEATURES))
            mlflow.log_param("dataset_sha256", dataset_hash)

            mlflow.log_metric("validation_rmse", rmse)
            mlflow.log_metric("validation_mae", mae)
            mlflow.log_metric("validation_r2", r2)

            mlflow.log_dict({
                "dataset_sha256": dataset_hash,
                "dataset_rows": len(df),
                "train_rows": len(X_train),
                "validation_rows": len(X_validation),
                "test_rows": len(X_test),
                "features": FEATURES,
                "target": TARGET,
                "split_strategy": "chronological_64_16_20"
            }, "dataset_metadata.json")

            results.append({
                "run_id": run.info.run_id,
                "name": config["name"],
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "params": config.copy()
            })

            print(f"[{index}/{len(CONFIGS)}] {config['name']}")
            print(f"  Validation RMSE: {rmse:.6f}")
            print(f"  Validation MAE : {mae:.6f}")
            print(f"  Validation R2  : {r2:.6f}")
            print()

    # Pilih model berdasarkan validation RMSE.
    best = min(results, key=lambda item: item["rmse"])

    print("=== BEST MODEL FROM VALIDATION ===")
    print(f"Nama model : {best['name']}")
    print(f"RMSE       : {best['rmse']:.6f}")
    print(f"Parameter  : {best['params']}")

    # ========================================================
    # 4. REFIT MODEL TERBAIK DENGAN TRAIN + VALIDATION
    # ========================================================

    final_model = RandomForestRegressor(
        n_estimators=best["params"]["n_estimators"],
        max_depth=best["params"]["max_depth"],
        min_samples_leaf=best["params"]["min_samples_leaf"],
        min_samples_split=best["params"]["min_samples_split"],
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X_train_validation, y_train_validation)

    model_path = MODEL_DIR / "best_gold_price_model.joblib"
    joblib.dump(final_model, model_path)

    # Simpan run ID dari eksperimen terpilih.
    metadata = {
        "best_run_id": best["run_id"],
        "best_experiment": best["name"],
        "best_parameters": best["params"],
        "selection_metric": "validation_rmse",
        "validation_rmse": best["rmse"],
        "validation_mae": best["mae"],
        "validation_r2": best["r2"],
        "target": TARGET,
        "features": FEATURES,
        "dataset_path": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": dataset_hash,
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "validation_rows": len(X_validation),
        "train_validation_rows": len(X_train_validation),
        "test_rows": len(X_test),
        "split_strategy": "chronological_64_16_20",
        "test_set_used_for_selection": False,
        "model_file": str(model_path.relative_to(ROOT)),
        "mlflow_experiment": EXPERIMENT_NAME
    }

    metadata_path = MODEL_DIR / "best_model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # Catat model final yang sudah dilatih ulang ke MLflow.
    with mlflow.start_run(run_name="best_model_refit_train_validation") as run:
        mlflow.log_params(best["params"])
        mlflow.log_param("training_strategy", "refit_train_validation")
        mlflow.log_param("dataset_sha256", dataset_hash)
        mlflow.log_metric("selected_validation_rmse", best["rmse"])

        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        mlflow.log_artifact(str(metadata_path))

    print()
    print("=== FINAL MODEL SAVED ===")
    print(f"Model       : {model_path.relative_to(ROOT)}")
    print(f"Metadata    : {metadata_path.relative_to(ROOT)}")
    print(f"Train+Valid : {len(X_train_validation)} baris")
    print(f"Test        : {len(X_test)} baris (belum digunakan)")
    print("Training selesai. Evaluasi akhir dilakukan terpisah.")


if __name__ == "__main__":
    main()
