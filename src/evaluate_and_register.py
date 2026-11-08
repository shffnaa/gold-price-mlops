import hashlib
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

MAX_ACCEPTABLE_RMSE = 0.95
MIN_RELATIVE_IMPROVEMENT = 0.02


# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def evaluate_quality_gates(rmse, baseline_rmse=None):
    absolute_gate_passed = rmse <= MAX_ACCEPTABLE_RMSE

    required_rmse = None

    if baseline_rmse is None:
        relative_gate_status = "NOT_EVALUATED"
    else:
        required_rmse = baseline_rmse * (1 - MIN_RELATIVE_IMPROVEMENT)
        relative_gate_status = (
            "PASS" if rmse < required_rmse else "FAIL"
        )

    passed = (
        absolute_gate_passed
        and relative_gate_status == "PASS"
    )

    if not absolute_gate_passed:
        status = "FAIL"
    elif relative_gate_status == "NOT_EVALUATED":
        status = "BLOCKED"
    else:
        status = "PASS" if passed else "FAIL"

    return {
        "absolute_gate_passed": absolute_gate_passed,
        "relative_gate_status": relative_gate_status,
        "baseline_rmse": baseline_rmse,
        "required_rmse": required_rmse,
        "passed": passed,
        "status": status,
    }


# =====================================================
# 3. MAIN EVALUATION
# =====================================================

def main():
    print("=== AUTOMATED MODEL EVALUATION ===")

    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    # Pastikan dataset sama dengan dataset saat training.
    current_dataset_hash = sha256_file(DATA_PATH)

    if current_dataset_hash != metadata["dataset_sha256"]:
        raise ValueError(
            "Dataset evaluation berbeda dengan dataset training!"
        )

    print("Dataset integrity check: PASSED")

    # Pastikan skema split sesuai dengan training terbaru.
    if metadata.get("split_strategy") != "chronological_64_16_20":
        raise ValueError("Strategi pembagian data tidak sesuai!")

    # Urutan data harus sama dengan proses training.
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    features = metadata["features"]
    target = "target_next_close_usd_per_gram"

    df[target] = df["Close_USD_per_gram"].shift(-1)
    df = df.dropna(subset=features + [target]).copy()

    # Training: 64%, validation: 16%, testing: 20%.
    # Test dimulai setelah 80% data pertama.
    validation_end = int(len(df) * 0.80)
    test = df.iloc[validation_end:]

    X_test = test[features]
    y_test = test[target]

    # Verifikasi jumlah data test dengan metadata training.
    if len(X_test) != metadata["test_rows"]:
        raise ValueError(
            f"Jumlah data test tidak sesuai! "
            f"Metadata: {metadata['test_rows']}, "
            f"aktual: {len(X_test)}"
        )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    # =================================================
    # 4. HITUNG METRIK
    # =================================================

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    print(f"RMSE       : {rmse:.6f}")
    print(f"MAE        : {mae:.6f}")
    print(f"R2         : {r2:.6f}")
    print(f"Data test  : {len(test)} baris")
    print(f"Threshold  : RMSE <= {MAX_ACCEPTABLE_RMSE}")
    print("Split      : Chronological 64:16:20")

    # =================================================
    # 5. QUALITY GATE
    # =================================================

    gate_results = evaluate_quality_gates(rmse)

    absolute_gate_passed = gate_results["absolute_gate_passed"]
    relative_gate_status = gate_results["relative_gate_status"]
    baseline_rmse = gate_results["baseline_rmse"]
    required_rmse = gate_results["required_rmse"]
    passed = gate_results["passed"]
    status = gate_results["status"]

    print(f"\nEvaluation status: {status}")

    # =================================================
    # 6. MLFLOW
    # =================================================

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()
    registered_version = None

    with mlflow.start_run(run_name="lk08_automated_evaluation") as run:

        mlflow.log_metrics({
            "evaluation_rmse": rmse,
            "evaluation_mae": mae,
            "evaluation_r2": r2,
        })

        mlflow.log_params({
            "evaluation_threshold_rmse": MAX_ACCEPTABLE_RMSE,
            "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
            "absolute_gate_status": (
                "PASS" if absolute_gate_passed else "FAIL"
            ),
            "relative_gate_status": relative_gate_status,
            "evaluation_status": status,
            "source_run_id": metadata["best_run_id"],
            "split_strategy": metadata["split_strategy"],
        })

        evaluation_run_id = run.info.run_id

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
            print("\nModel belum memenuhi seluruh quality gate.")
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
        "test_rows": len(test),
        "split_strategy": metadata["split_strategy"],
        "quality_gates": {
            "absolute_rmse": {
                "status": "PASS" if absolute_gate_passed else "FAIL",
                "actual": rmse,
                "maximum": MAX_ACCEPTABLE_RMSE,
            },
            "relative_improvement": {
                "status": relative_gate_status,
                "baseline_rmse": baseline_rmse,
                "required_rmse": required_rmse,
                "minimum_improvement": MIN_RELATIVE_IMPROVEMENT,
                "reason": (
                    "Baseline LK-06 belum tersedia untuk dibandingkan "
                    "pada dataset LK-08 yang sama."
                    if baseline_rmse is None
                    else "RMSE dibandingkan dengan baseline menggunakan "
                    "batas peningkatan minimal 2%."
                ),
            },
        },
        "evaluation_run_id": evaluation_run_id,
        "source_run_id": metadata["best_run_id"],
        "dataset_sha256": metadata["dataset_sha256"],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nLaporan tersimpan: {REPORT_PATH}")

    if passed:
        print("\nSUCCESS: Model memenuhi seluruh quality gate.")
        print("PIPELINE PASSED: Model berhasil masuk Staging.")
    else:
        print("\nWARNING: Model gagal memenuhi quality gate.")
        print("Evaluasi berhasil diselesaikan.")
        print("Registrasi model dilewati.")
        print("PIPELINE COMPLETED: Quality gate memblokir registrasi model.")


if __name__ == "__main__":
    main()
