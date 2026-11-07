import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# 1. KONFIGURASI
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "gold_features_latest.csv"
MODEL_PATH = ROOT / "models" / "best_gold_price_model.joblib"
METADATA_PATH = ROOT / "models" / "best_model_metadata.json"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "evaluation.json"

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "gold-price-forecasting"
MODEL_NAME = "GoldPriceForecasting"

# Threshold berdasarkan LK-01
MAX_ACCEPTABLE_RMSE = 0.95


# =====================================================
# 2. LOAD DATA DAN MODEL
# =====================================================

def main():
    print("=== AUTOMATED MODEL EVALUATION ===")

    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    df = pd.read_csv(DATA_PATH)

    features = metadata["features"]
    target = "target_next_close_usd_per_gram"

    df[target] = df["Close_USD_per_gram"].shift(-1)
    df = df.dropna(subset=features + [target]).copy()

    split_index = int(len(df) * 0.8)
    test = df.iloc[split_index:]

    X_test = test[features]
    y_test = test[target]

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    # =================================================
    # 3. HITUNG METRIK
    # =================================================

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    print(f"RMSE       : {rmse:.6f}")
    print(f"MAE        : {mae:.6f}")
    print(f"R2         : {r2:.6f}")
    print(f"Data test  : {len(test)} baris")
    print(f"Threshold  : RMSE <= {MAX_ACCEPTABLE_RMSE}")

    # =================================================
    # 4. EVALUASI PASS / FAIL
    # =================================================

    passed = rmse <= MAX_ACCEPTABLE_RMSE

    status = "PASS" if passed else "FAIL"

    print(f"\nEvaluation status: {status}")

    # =================================================
    # 5. SIAPKAN DATABASE MLFLOW
    # =================================================

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()

    registered_version = None

    # Catat hasil evaluasi, termasuk jika model gagal.
    with mlflow.start_run(run_name="lk08_automated_evaluation") as run:

        mlflow.log_metrics({
            "evaluation_rmse": rmse,
            "evaluation_mae": mae,
            "evaluation_r2": r2,
        })

        mlflow.log_params({
            "evaluation_threshold_rmse": MAX_ACCEPTABLE_RMSE,
            "evaluation_status": status,
            "source_run_id": metadata["best_run_id"],
        })

        evaluation_run_id = run.info.run_id

        # =============================================
        # 6. REGISTRASI HANYA JIKA PASS
        # =============================================

        if passed:
            print("\nModel memenuhi threshold. Registrasi dimulai.")

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )

            model_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=MODEL_NAME,
            )

            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=model_version.version,
                stage="Staging",
            )

            registered_version = model_version.version

            print(f"Model version : {registered_version}")
            print("Model stage   : Staging")

        else:
            print("\nModel TIDAK memenuhi threshold.")
            print("Registrasi ke Model Registry dilewati.")

    # =================================================
    # 7. SIMPAN LAPORAN EVALUASI
    # =================================================

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "status": status,
        "registered": passed,
        "model_name": MODEL_NAME if passed else None,
        "model_version": registered_version,
        "model_stage": "Staging" if passed else None,
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        },
        "thresholds": {
            "max_acceptable_rmse": MAX_ACCEPTABLE_RMSE,
        },
        "evaluation_run_id": evaluation_run_id,
        "source_run_id": metadata["best_run_id"],
        "dataset_sha256": metadata["dataset_sha256"],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nLaporan tersimpan: {REPORT_PATH}")

    # Exit code 1 membuat pipeline CI gagal jika evaluasi gagal.
    if not passed:
        print("\nPIPELINE FAILED: Model belum memenuhi kriteria LK-01.")
        sys.exit(1)

    print("\nPIPELINE PASSED: Model berhasil masuk Staging.")


if __name__ == "__main__":
    main()
