import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


LEGACY_MODEL_PATH = (
    Path("/legacy_mlruns/1/models/")
    / "m-7607cb2e75d446049d509b7e4b4aa301"
    / "artifacts"
)

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "GoldPriceForecasting")
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-server:5000",
)


def main():
    print("===== LK-08 TO LK-09 MODEL MIGRATION =====")
    print(f"Tracking URI : {TRACKING_URI}")
    print(f"Model name   : {MODEL_NAME}")
    print(f"Source       : {LEGACY_MODEL_PATH}")

    if not LEGACY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Legacy model artifact not found: {LEGACY_MODEL_PATH}"
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Avoid duplicate registration when the container restarts.
    try:
        registered_model = client.get_registered_model(MODEL_NAME)
        versions = client.search_model_versions(
            filter_string=f"name='{MODEL_NAME}'"
        )

        if versions:
            print(
                f"Model already registered. "
                f"Existing versions: {len(versions)}"
            )
            print("Migration skipped.")
            return

    except MlflowException:
        registered_model = None

    print("\nLoading original LK-08 model...")
    model = mlflow.sklearn.load_model(str(LEGACY_MODEL_PATH))

    mlflow.set_experiment("gold-price-forecasting-lk09")

    print("Logging model to the new MLflow server...")

    with mlflow.start_run(run_name="lk08_model_migration") as run:
        mlflow.log_params({
            "migration_source": "LK08",
            "source_model_id": "m-7607cb2e75d446049d509b7e4b4aa301",
            "source_run_id": "dbc961c710e34a4db6979ef96fb485a7",
            "serialization_format": "skops",
        })

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"Migration run ID: {run.info.run_id}")

    print("\nRegistering model...")

    model_version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=MODEL_NAME,
    )

    print("\n===== MIGRATION SUCCESS =====")
    print(f"Model name    : {MODEL_NAME}")
    print(f"Model version : {model_version.version}")
    print(f"Model status  : {model_version.status}")


if __name__ == "__main__":
    main()
