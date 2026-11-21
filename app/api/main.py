import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-server:5000",
)

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "GoldPriceForecasting",
)

MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION", "1")

model = None

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


class PredictionInput(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Close_USD_per_gram: float
    is_outlier_shock: int
    SMA_7: float
    SMA_30: float
    Rolling_Std_30: float
    pct_change: float
    RSI_14: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    mlflow.set_tracking_uri(TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

    print(f"Loading model from: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    print("Model loaded successfully.")

    yield

    model = None


app = FastAPI(
    title="Gold Price Forecasting API",
    description="Gold price prediction using the registered MLflow model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "Gold Price Forecasting API",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tracking_uri": TRACKING_URI,
    }


@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
        input_data = pd.DataFrame(
            [data.model_dump()],
            columns=FEATURES,
        )

        prediction = model.predict(input_data)

        return {
            "prediction": float(prediction[0]),
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "unit": "USD per gram",
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )
