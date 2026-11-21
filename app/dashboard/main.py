import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Gold Price Forecasting",
    page_icon="🥇",
    layout="wide",
)

st.title("🥇 Gold Price Forecasting")
st.caption("Gold price prediction powered by Machine Learning, FastAPI, and MLflow")

# API connection status
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    response.raise_for_status()
    health = response.json()

    if health.get("model_loaded"):
        st.success("API connected — Model loaded successfully")
    else:
        st.warning("API connected, but model is not loaded yet.")

except requests.RequestException:
    st.error("Cannot connect to the prediction API.")

st.divider()

st.subheader("Prediction Input")
st.write(
    "Masukkan 12 fitur hasil preprocessing sesuai urutan fitur model."
)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        open_price = st.number_input("Open", value=0.0)
        high = st.number_input("High", value=0.0)
        low = st.number_input("Low", value=0.0)
        close = st.number_input("Close", value=0.0)

    with col2:
        volume = st.number_input("Volume", value=0.0)
        close_usd = st.number_input("Close USD per gram", value=0.0)
        outlier = st.selectbox("Outlier Shock", options=[0, 1])
        sma_7 = st.number_input("SMA 7", value=0.0)

    with col3:
        sma_30 = st.number_input("SMA 30", value=0.0)
        rolling_std = st.number_input("Rolling Std 30", value=0.0)
        pct_change = st.number_input("Percentage Change", value=0.0)
        rsi = st.number_input("RSI 14", value=0.0)

    submitted = st.form_submit_button(
        "Predict Gold Price",
        use_container_width=True,
    )

if submitted:
    payload = {
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "Close_USD_per_gram": close_usd,
        "is_outlier_shock": outlier,
        "SMA_7": sma_7,
        "SMA_30": sma_30,
        "Rolling_Std_30": rolling_std,
        "pct_change": pct_change,
        "RSI_14": rsi,
    }

    try:
        with st.spinner("Processing prediction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

        st.success("Prediction completed!")

        st.metric(
            label="Predicted Gold Price",
            value=f"${result['prediction']:,.4f} / gram",
        )

        st.caption(
            f"Model: {result['model_name']} "
            f"| Version: {result['model_version']}"
        )

    except requests.RequestException as error:
        st.error(f"Prediction request failed: {error}")

st.divider()

st.caption(
    "Academic MLOps Project — Gold Price Forecasting | "
    "Streamlit + FastAPI + MLflow + PostgreSQL"
)
