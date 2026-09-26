import os
import glob
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TROY_OUNCE_TO_GRAM = 31.1034768


def get_latest_raw_file(raw_dir: str = RAW_DIR) -> str:
    """Mengambil file CSV mentah yang paling baru berdasarkan timestamp."""
    csv_files = glob.glob(os.path.join(raw_dir, "gold_raw_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Tidak ditemukan file gold_raw_*.csv di direktori {raw_dir}")
    latest_file = max(csv_files, key=os.path.getctime)
    return latest_file


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan format, deduplikasi, forward-fill, dan konversi satuan."""
    initial_rows = len(df)

    # 1. Validasi & Standarisasi Tanggal
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by="Date", inplace=True)

    # 2. Deduplikasi Baris
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    dedup_rows = initial_rows - len(df)
    if dedup_rows > 0:
        logging.info(f"Deduplikasi: {dedup_rows} baris duplikat dihapus.")

    # 3. Penanganan Missing Value (Forward-fill untuk non-trading days bursa)
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # 4. Konversi Satuan ke USD/gram
    df["Close_USD_per_gram"] = df["Close"] / TROY_OUNCE_TO_GRAM

    # 5. Outlier Flagging (> +-15% daily change tidak dihapus, hanya diberi flag)
    pct_change = df["Close_USD_per_gram"].pct_change()
    df["is_outlier_shock"] = pct_change.abs() > 0.15
    outlier_count = df["is_outlier_shock"].sum()
    logging.info(
        f"Pembersihan selesai: {outlier_count} guncangan ekstrem ditandai sebagai sinyal geopolitik.")

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung indikator teknikal: SMA, RSI, Volatilitas, dan Trend Label."""
    # Simple Moving Average
    df["SMA_7"] = df["Close_USD_per_gram"].rolling(
        window=7, min_periods=1).mean()
    df["SMA_30"] = df["Close_USD_per_gram"].rolling(
        window=30, min_periods=1).mean()

    # Rolling Standard Deviation (30-day volatility)
    df["Rolling_Std_30"] = df["Close_USD_per_gram"].rolling(
        window=30, min_periods=1).std().fillna(0)

    # Persentase Perubahan Harian
    df["pct_change"] = df["Close_USD_per_gram"].pct_change().fillna(0)

    # Relative Strength Index (RSI 14)
    delta = df["Close_USD_per_gram"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Target Labeling BUY/HOLD/SELL
    def assign_label(val):
        if val > 0.005:
            return "BUY"
        elif val < -0.005:
            return "SELL"
        return "HOLD"

    df["label_trend"] = df["pct_change"].apply(assign_label)
    return df


def run_preprocessing():
    raw_path = get_latest_raw_file()
    logging.info(f"Memproses berkas mentah: {raw_path}")
    df_raw = pd.read_csv(raw_path)

    df_clean = clean_data(df_raw)
    df_final = feature_engineering(df_clean)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_parquet = os.path.join(PROCESSED_DIR, "gold_features_latest.parquet")
    out_csv = os.path.join(PROCESSED_DIR, "gold_features_latest.csv")

    df_final.to_parquet(out_parquet, index=False)
    df_final.to_csv(out_csv, index=False)

    logging.info(
        f"Prapemrosesan berhasil. Data siap pakai disimpan ke:\n - {out_parquet}\n - {out_csv}")
    logging.info(f"Total baris siap training/DVC: {len(df_final)}")


if __name__ == "__main__":
    run_preprocessing()
