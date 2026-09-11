# MLOps - Peramalan Harga Emas dengan Deteksi Data Drift dan Continuous Training

## Tujuan Proyek

Membangun pipeline MLOps end-to-end untuk memprediksi harga penutupan emas harian (GC=F, COMEX) dalam satuan USD/gram, dilengkapi mekanisme deteksi data drift (KS-Test) dan continuous training otomatis untuk menjaga akurasi model tetap relevan terhadap dinamika pasar.


## Struktur Direktori

```text
gold-price-mlops/
├── data/
│   ├── raw/           # Data mentah hasil ingestion dari Yahoo Finance
│   └── processed/     # Data yang sudah melalui feature engineering
├── models/            # Model artifacts & checkpoints
├── src/               # Source code (ingestion, training, inference, drift detection)
├── config/            # File konfigurasi (parameter model, threshold, dll)
├── notebooks/         # Eksplorasi data & eksperimen awal
├── tests/             # Unit test
├── docs/              # Dokumentasi tambahan
├── .devcontainer/     # Konfigurasi GitHub Codespaces
├── .gitignore         # File pengabaian git
├── requirements.txt   # Dependensi pustaka Python
└── README.md          # Dokumentasi proyek
```

## Cara Menjalankan (via GitHub Codespaces)

1. Klik tombol **Code** → tab **Codespaces** → **Create codespace on main**.
2. Tunggu proses build environment selesai (2–3 menit). Dependencies pada
   `requirements.txt` akan otomatis terinstall lewat `postCreateCommand`.
3. Verifikasi environment:
```bash
   python --version
   pip list
```
4. Jalankan notebook eksplorasi awal di folder `notebooks/`.

## Cara Menjalankan (Lokal)

```bash
git clone https://github.com/shffnaa/gold-price-mlops.git
cd gold-price-mlops
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Branching Strategy

Proyek ini mengikuti **GitHub Flow**:
- Branch `main` selalu dalam kondisi stabil dan production-ready.
- Setiap fitur/eksperimen baru dikembangkan di branch terpisah, contoh:
  `feat/initial-eda`, `feat/drift-detection`, `feat/model-training`.
- Perubahan diajukan melalui Pull Request dan direview sebelum di-merge
  ke `main`.
