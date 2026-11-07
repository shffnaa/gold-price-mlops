import pandas as pd
from src.preprocess import clean_data, feature_engineering


def test_clean_data_converts_close_to_usd_per_gram():
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02"],
        "Open": [100, 101],
        "High": [102, 103],
        "Low": [99, 100],
        "Close": [3100, 3200],
        "Volume": [1000, 1100],
    })

    result = clean_data(df)

    assert "Close_USD_per_gram" in result.columns
    assert result["Close_USD_per_gram"].iloc[0] > 0


def test_feature_engineering_creates_indicators():
    df = pd.DataFrame({
        "Close_USD_per_gram": [100 + i for i in range(35)]
    })

    result = feature_engineering(df)

    expected_columns = [
        "SMA_7",
        "SMA_30",
        "Rolling_Std_30",
        "pct_change",
        "RSI_14",
        "label_trend",
    ]

    for column in expected_columns:
        assert column in result.columns

    assert result["label_trend"].notna().all()
