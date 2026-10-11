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

## Data Acquisition & Preprocessing 

### Menjalankan Skrip Pengumpul Data (`src/ingest_data.py`)

Skrip ini mengambil data harga emas (ticker `GC=F`) dari Yahoo Finance API
melalui pustaka `yfinance`, sesuai sumber data yang direncanakan pada LK-03.

```bash
# Sekali di awal proyek: ambil histori ~2 tahun (agar dataset awal >= 500 baris)
python src/ingest_data.py --full-history

# Update harian rutin (dipanggil oleh scheduler / GitHub Actions)
python src/ingest_data.py

# Rentang waktu kustom
python src/ingest_data.py --period 5d
```

**Perilaku penting:**
- **Non-destructive**: setiap eksekusi membuat berkas baru bertimestamp di
  `data/raw/gold_raw_<YYYYMMDD_HHMMSS>.csv` — data lama tidak pernah tertimpa.
- **Resilien terhadap gangguan jaringan**: retry otomatis hingga 3x dengan
  *exponential backoff* jika Yahoo Finance API gagal merespons.
- **Data lineage**: setiap berkas CSV disertai `<nama_file>.meta.json` yang
  mencatat `source_url`, `accessed_at`, `period_requested`, dan `record_count`
  — dipakai sebagai dasar audit trail untuk implementasi DVC di LK-05.

### Menjalankan Skrip Prapemrosesan (`src/preprocess.py`)

Skrip ini membersihkan berkas mentah terbaru (atau seluruhnya) dari `data/raw/`
dan menyimpan hasilnya ke `data/processed/`, siap untuk tahap ekstraksi fitur.

```bash
# Proses hanya berkas raw TERBARU
python src/preprocess.py

# Gabungkan & proses SEMUA berkas raw yang ada
python src/preprocess.py --all
```

**Langkah pembersihan yang dilakukan:**
1. Validasi skema (kolom wajib: `Date, Open, High, Low, Close, Volume, Ticker`)
2. Deduplikasi baris dengan tanggal yang sama
3. Penanganan *missing values* (forward-fill untuk hari libur bursa)
4. Konversi satuan harga USD/troy-ounce → USD/gram (`Close_USD_per_gram`)
5. Penandaan (bukan penghapusan) lonjakan harga >15% sebagai potensi
   *geopolitical drift* (lihat LK-01 BAB II)

Output: `data/processed/gold_clean_<YYYYMMDD_HHMMSS>.csv`

### Format Output Data

| Berkas | Lokasi | Format | Keterangan |
|---|---|---|---|
| Data mentah | `data/raw/gold_raw_*.csv` | CSV | Hasil langsung `ingest_data.py`, skema OHLCV |
| Metadata lineage | `data/raw/*.meta.json` | JSON | Source, waktu akses, jumlah baris |
| Data bersih | `data/processed/gold_clean_*.csv` | CSV | Hasil `preprocess.py`, siap ekstraksi fitur |

### Dependencies

```bash
pip install -r requirements.txt
```

Lihat `requirements.txt` untuk daftar lengkap (`yfinance`, `pandas`, `numpy`, `requests`).

## Data Versioning with DVC

### Tujuan

Proyek ini menggunakan **Data Version Control (DVC)** untuk melacak perubahan dataset mentah harga emas yang diperoleh dari Yahoo Finance. DVC membantu menjaga riwayat dataset agar perubahan data dapat ditelusuri dan digunakan kembali dalam eksperimen machine learning.

### 1. Inisialisasi DVC

DVC diinisialisasi satu kali pada repository proyek menggunakan perintah berikut:

```bash
dvc init
```

Konfigurasi DVC kemudian disimpan ke Git:

```bash
git add .dvc .dvcignore .gitignore
git commit -m "chore: initialize DVC"
```

### 2. Mengambil dan Melacak Dataset Awal

Dataset diperoleh melalui script ingestion:

```bash
python src/ingest_data.py
```

Script mengambil data harga emas dengan ticker `GC=F` dari Yahoo Finance. Setiap eksekusi menghasilkan file CSV bertimestamp dan file metadata lineage dalam folder `data/raw/`.

Dataset kemudian dilacak menggunakan DVC:

```bash
dvc add data/raw
git add data/raw.dvc data/.gitignore
git commit -m "feat: track initial raw dataset with DVC"
```

File `data/raw.dvc` berfungsi sebagai pointer metadata versi dataset yang dilacak Git, sedangkan isi dataset dikelola oleh DVC.

### 3. Memperbarui Versi Dataset

Untuk membuat versi dataset berikutnya, jalankan ingestion kembali:

```bash
python src/ingest_data.py
```

Setelah file baru terbentuk, perbarui pelacakan DVC:

```bash
dvc add data/raw
git add data/raw.dvc data/.gitignore
git commit -m "feat: update raw dataset version with DVC"
```

Setiap proses ingestion menghasilkan nama file berdasarkan timestamp. Dengan demikian, file dari proses sebelumnya tetap tersedia untuk dibandingkan dengan file yang lebih baru.

### 4. Memeriksa Perubahan Dataset

Gunakan perintah berikut untuk memeriksa konsistensi dataset dan membandingkan versi:

```bash
dvc status
dvc diff HEAD~1 HEAD
git log --oneline -5
```

- `dvc status` memeriksa apakah kondisi data lokal sesuai dengan metadata DVC.
- `dvc diff HEAD~1 HEAD` membandingkan perubahan dataset di antara dua commit Git.
- `git log --oneline -5` menampilkan riwayat commit terbaru yang mencatat perubahan pointer dataset.

### 5. Pentingnya Data Versioning

Data harga emas bersifat dinamis sehingga hasil ingestion dapat berubah dari waktu ke waktu. Tanpa pencatatan versi, eksperimen yang dilakukan pada waktu berbeda dapat menggunakan dataset yang berbeda tanpa disadari.

DVC memungkinkan versi dataset ditelusuri melalui metadata dan riwayat commit Git. Hal ini mendukung reproduksibilitas eksperimen, audit perubahan data, serta analisis perbedaan hasil pelatihan model. Ketika prediksi model berubah, versi dataset, kode, konfigurasi, dan parameter pelatihan dapat diperiksa untuk membantu menelusuri penyebabnya.

### 6. Preprocessing Data

Sebelum digunakan untuk pemodelan, data mentah dapat diproses menggunakan script `src/preprocess.py` sesuai tahapan preprocessing yang tersedia pada proyek, yaitu validasi skema, deduplikasi tanggal, penanganan missing values, konversi satuan harga, dan penandaan potensi lonjakan harga.

Perintah untuk memproses dataset mentah terbaru:

```bash
python src/preprocess.py
```

Untuk memproses seluruh file dataset mentah yang tersedia:

```bash
python src/preprocess.py --all
```

Hasil preprocessing disimpan di folder `data/processed/` sesuai implementasi script.

### 7. Hasil Praktikum

Pada praktikum, DVC berhasil diinisialisasi dan digunakan untuk melacak dataset mentah. Dua kali proses ingestion menghasilkan file CSV dan metadata lineage dengan timestamp berbeda. Perintah `dvc diff HEAD~1 HEAD` menunjukkan dua file baru pada versi kedua, sedangkan `dvc status` menunjukkan bahwa data lokal sesuai dengan metadata DVC pada saat pemeriksaan.

Hasil tersebut membuktikan adanya perubahan berkas yang tercatat dalam versioning. Namun, penambahan file baru tidak otomatis berarti jumlah observasi unik bertambah atau nilai harga emas berubah.