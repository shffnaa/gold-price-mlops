import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
import pandas as pd
import yfinance as yf

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

TICKER_SYMBOL = "GC=F"
SOURCE_URL = f"https://query1.finance.yahoo.com/v8/finance/chart/{TICKER_SYMBOL}"
OUTPUT_DIR = "data/raw"


def fetch_gold_data_with_retry(period: str = "2y", interval: str = "1d", max_retries: int = 3) -> pd.DataFrame:
    """
    Mengambil data pasar dari Yahoo Finance dengan mekanisme exponential backoff.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(
                f"Mencoba mengambil data {TICKER_SYMBOL} (Percobaan {attempt}/{max_retries})...")
            df = yf.download(tickers=TICKER_SYMBOL, period=period,
                             interval=interval, progress=False)

            if df.empty:
                raise ValueError("Data yang diterima kosong.")

            # Format multi-index handling jika ada dari yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)
            df["Ticker"] = TICKER_SYMBOL
            logging.info(f"Ekstraksi berhasil: {len(df)} baris data diterima.")
            return df

        except Exception as e:
            wait_time = 2 ** attempt
            logging.warning(
                f"Gagal mengambil data pada percobaan {attempt}: {e}")
            if attempt < max_retries:
                logging.info(
                    f"Menunggu {wait_time} detik sebelum mencoba ulang...")
                time.sleep(wait_time)
            else:
                logging.error("Seluruh percobaan pengambilan data gagal.")
                raise e


def save_non_destructive(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """
    Menyimpan data dengan penamaan timestamp dan mencatat metadata lineage.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename_csv = f"gold_raw_{ts}.csv"
    filepath_csv = os.path.join(output_dir, filename_csv)

    filename_meta = f"gold_raw_{ts}.meta.json"
    filepath_meta = os.path.join(output_dir, filename_meta)

    # 1. Simpan berkas data mentah
    df.to_csv(filepath_csv, index=False)
    logging.info(f"Dataset mentah disimpan ke: {filepath_csv}")

    # 2. Simpan metadata lineage dasar (Slide 13 Modul Ingestion)
    metadata = {
        "ticker": TICKER_SYMBOL,
        "source_url": SOURCE_URL,
        "accessed_at": datetime.now().isoformat(),
        "record_count": len(df),
        "columns": list(df.columns),
        "http_status": 200,
        "ingestion_mode": "non-destructive"
    }
    with open(filepath_meta, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata lineage disimpan ke: {filepath_meta}")

    return filepath_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script Ingestion Data Harga Emas")
    parser.add_argument("--full-history", action="store_true",
                        help="Ambil 2 tahun historis (>= 500 baris untuk DVC baseline)")
    args = parser.parse_args()

    fetch_period = "2y" if args.full_history else "1mo"
    logging.info(f"Menjalankan ingestion mode: period={fetch_period}")

    data = fetch_gold_data_with_retry(period=fetch_period)
    save_non_destructive(data)
