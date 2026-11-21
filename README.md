# MLOps - Peramalan Harga Emas dengan Deteksi Data Drift dan Continuous Training

## Tujuan Proyek

Membangun pipeline MLOps end-to-end untuk memprediksi harga penutupan emas harian (GC=F, COMEX) dalam satuan USD/gram, dilengkapi mekanisme deteksi data drift (KS-Test) dan continuous training otomatis untuk menjaga akurasi model tetap relevan terhadap dinamika pasar.


## Struktur Direktori
```text
gold-price-mlops/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── best_model_metadata.json
│   └── best_gold_price_model.joblib  # lokal, diabaikan Git
├── src/
│   ├── ingest_data.py
│   ├── preprocess.py
│   └── train.py
├── mlflow.db                         # lokal, diabaikan Git
├── data/raw.dvc
├── requirements.txt
└── README.md
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

---

## LK-06 — MLflow Experiment Tracking 

### Tujuan

Pada tahap LK-06, proyek mengimplementasikan manajemen eksperimen machine learning menggunakan **MLflow** untuk mencatat parameter, metrik evaluasi, dan artefak model. Hasil setiap eksperimen dapat dibandingkan untuk menentukan konfigurasi model terbaik sebelum digunakan pada tahap deployment berikutnya.

### 1. Menjalankan Training

Pastikan dataset hasil preprocessing tersedia di `data/processed/gold_features_latest.csv`. Jalankan training dengan perintah:

```bash
python src/preprocess.py
python src/train.py
```

Skrip `src/train.py` menjalankan sepuluh konfigurasi Random Forest terbaru menggunakan pembagian data secara kronologis, yaitu 400 baris untuk training dan 101 baris untuk testing.

### 2. Integrasi MLflow

Eksperimen dicatat pada experiment `gold-price-forecasting`. MLflow digunakan untuk:

- Mencatat hyperparameter setiap run menggunakan `mlflow.log_params()`.
- Mencatat metrik evaluasi menggunakan `mlflow.log_metric()`.
- Menyimpan model sebagai artefak menggunakan `mlflow.sklearn.log_model()`.
- Membandingkan hasil eksperimen melalui MLflow UI.
- Mendaftarkan model terpilih pada Model Registry.

Pada saat pengujian, MLflow menampilkan **13 runs**, yang terdiri dari sepuluh run terbaru dan tiga run historis.

### 3. Membuka MLflow UI

Jalankan perintah berikut dari root repository:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Pada GitHub Codespaces, buka tab **Ports**, lalu akses port `5000`. Di halaman experiment `gold-price-forecasting`, hasil run dan metrik dapat diperiksa serta dibandingkan.

### 4. Hasil Eksperimen

Konfigurasi terbaik dari sepuluh run terbaru adalah `rf_trees_300`, berdasarkan nilai test RMSE terendah.

| Parameter / Metrik | Hasil |
|---|---:|
| `n_estimators` | 300 |
| `max_depth` | 10 |
| `min_samples_leaf` | 2 |
| `min_samples_split` | 2 |
| Test RMSE | 3.8682 USD/gram |
| Test MAE | 3.2219 USD/gram |
| Test R² | 0.5871 |

RMSE dan MAE mengukur kesalahan prediksi; nilai yang lebih rendah menunjukkan kesalahan yang lebih kecil. R² menggambarkan kemampuan model menjelaskan variasi target.

Metrik tersebut digunakan untuk membandingkan konfigurasi dan memilih model pada eksperimen ini. Karena test set juga digunakan untuk seleksi konfigurasi, evaluasi independen pada data uji terpisah tetap diperlukan sebelum model digunakan di lingkungan produksi.

### 5. Model Registry

Model terpilih berhasil didaftarkan pada MLflow Model Registry dengan informasi berikut:

- **Registered model:** `GoldPriceForecasting`
- **Version:** `2`
- **Source run:** `rf_trees_300`

Registrasi ini mendokumentasikan versi model yang dipilih untuk tahap deployment berikutnya. Registrasi model tidak dengan sendirinya berarti model sudah di-deploy ke produksi.

### 6. Model Metadata

Metadata model terbaik disimpan pada:

```text
models/best_model_metadata.json
```

File tersebut mendokumentasikan run ID, parameter terbaik, metrik evaluasi, target prediksi, daftar fitur, lokasi dataset, jumlah baris data, dan hash SHA-256 dataset. Metadata ini dapat digunakan untuk membantu pelacakan dan reproduksi konfigurasi model.

Model lokal disimpan pada:

```text
models/best_gold_price_model.joblib
```

File model lokal dan database tracking `mlflow.db` dikecualikan dari Git melalui `.gitignore`. Skrip training dan file metadata JSON tetap disimpan dalam repository.

### 7. Laporan Praktikum

Laporan teknis LK-06 mendokumentasikan hasil eksperimen dan bukti dari MLflow UI, meliputi:

1. Daftar experiment runs.
2. Grafik perbandingan metrik.
3. Halaman Model Registry yang menampilkan `GoldPriceForecasting` Version 2.




## LK-07 — Model Registry, Versioning, dan Kesiapan Inferensi

### 1. Model Registry dan Versioning

Model terdaftar pada MLflow Model Registry dengan nama `GoldPriceForecasting`. Eksperimen LK-07 menghasilkan versi model baru sehingga riwayat versi dapat digunakan untuk menelusuri perubahan model.

Versi yang tercatat dalam registry:

| Versi | Source run | Test RMSE |
|---|---|---:|
| Version 1 | `rf_medium` | 3.8764 |
| Version 2 | `rf_trees_300` | 3.8682 |
| Version 3 | `lk07_rf_trees_250` | 3.8767 |

Version 2 dipilih sebagai model aktif karena memiliki test RMSE terendah di antara ketiga versi tersebut.

### 2. Alias Model Aktif

MLflow yang digunakan dalam praktikum ini mendukung alias model. Alias `staging` dan `production` diarahkan ke `GoldPriceForecasting` Version 2.

Metadata versi aktif disimpan pada:

```text
models/production_model.yaml
```

File tersebut mencatat nama model, alias production, versi model, source run ID, metrik evaluasi, referensi dataset, dan URI model registry.

### 3. Verifikasi Inferensi

Model production diuji dengan memuat model melalui URI alias:

```python
model = mlflow.pyfunc.load_model(
    "models:/GoldPriceForecasting@production"
)
```

Pengujian berhasil dilakukan menggunakan tiga baris input dari dataset fitur. Model menghasilkan tiga prediksi harga emas dalam USD/gram.

Keberhasilan pemuatan model dan keluaran prediksi menunjukkan bahwa versi yang dirujuk alias `production` dapat digunakan untuk inferensi. Hasil ini merupakan pengujian inferensi dasar, bukan bukti bahwa model telah di-deploy sebagai layanan produksi.

### 4. Catatan DVC dan Reproduksibilitas

Status DVC menunjukkan bahwa data dan pipeline yang dikonfigurasi berada dalam kondisi up to date pada lingkungan lokal. Namun, belum ada DVC remote yang dikonfigurasi, sehingga artefak atau cache DVC lokal belum disinkronkan ke penyimpanan remote.

Database tracking MLflow `mlflow.db` juga disimpan secara lokal dan dikecualikan dari Git. Untuk mereproduksi hasil pada lingkungan lain, konfigurasi tracking dan ketersediaan artefak model perlu disiapkan kembali.

### 5. Bukti Laporan LK-07

Laporan teknis LK-07 perlu menyertakan bukti berikut:

1. MLflow Model Registry yang menampilkan versi model.
2. Alias `staging` dan `production` yang mengarah ke Version 2.
3. Terminal yang menunjukkan inferensi production berhasil.
4. File `models/production_model.yaml` sebagai metadata versi model aktif.

---
## LK-08 — Code as a Trigger: MLOps Automation Pipeline

## 1. Tujuan

Pada tahap LK-08, proyek mengimplementasikan otomatisasi pipeline MLOps menggunakan GitHub Actions. Otomatisasi ini bertujuan menjalankan serangkaian proses machine learning ketika terdapat perubahan kode pada repository, sehingga proses pengujian, preprocessing, training, evaluasi, dan pengelolaan model dapat dilakukan secara terstruktur dan berulang.

Pipeline dirancang menggunakan pendekatan Code as a Trigger, yaitu perubahan kode yang di-push ke repository atau Pull Request dapat memicu workflow secara otomatis.

## 2. Konfigurasi GitHub Actions

Workflow otomatisasi disimpan pada:

`.github/workflows/mlops-automation.yaml`

Workflow menggunakan GitHub Actions dengan trigger:
- `push` ke branch `main`.
- `pull_request` ke branch `main`.

Ketika trigger terpenuhi, GitHub Actions menjalankan tahapan pipeline secara berurutan.

## 3. Tahapan Automation Pipeline

Pipeline MLOps terdiri dari beberapa tahapan berikut:

**a. Checkout Repository**

GitHub Actions mengambil source code dari repository agar seluruh proses dapat dijalankan pada environment runner.

**b. Setup Python Environment**

Workflow menyiapkan versi Python dan menginstal dependencies yang dibutuhkan oleh proyek melalui file requirements.

**c. Automated Testing**

Pipeline menjalankan pengujian otomatis menggunakan pytest untuk memeriksa fungsi preprocessing dan evaluasi model. Tahap ini membantu mendeteksi kesalahan kode sebelum proses training dan evaluasi dilanjutkan.

**d. Data Preprocessing**

Workflow menjalankan script preprocessing untuk mengolah dataset mentah menjadi dataset fitur yang siap digunakan dalam proses training.

**e. Model Training**

Workflow menjalankan script training untuk melatih beberapa konfigurasi Random Forest. Eksperimen training dicatat menggunakan MLflow sehingga parameter dan metrik setiap konfigurasi dapat dibandingkan.

**f. Model Evaluation dan Quality Gate**

Model dievaluasi menggunakan metrik RMSE, MAE, dan R². Hasil evaluasi diperiksa melalui mekanisme quality gate untuk menentukan apakah model memenuhi kriteria penerimaan.

Kriteria quality gate yang digunakan meliputi:
- RMSE maksimum sebesar 0.95 USD/gram.
- Peningkatan relatif minimal 2% dibandingkan baseline yang dapat dibandingkan secara valid.

Hasil evaluasi disimpan dalam file:

`reports/evaluation.json`

**g. Conditional Model Registration**

Registrasi model ke MLflow Model Registry dilakukan secara kondisional. Model hanya dapat didaftarkan apabila memenuhi kriteria quality gate yang ditentukan. Apabila model tidak memenuhi kriteria, proses registrasi dilewati agar model yang belum memenuhi standar tidak otomatis dipromosikan ke tahap Staging.

## 4. Dataset dan Data Versioning

Pipeline menggunakan dataset yang telah dikelola menggunakan Data Version Control (DVC). Metadata pelacakan dataset disimpan pada:

`data/raw.dvc`

Dataset mentah yang digunakan dalam implementasi LK-08 merupakan dataset yang telah dilacak dalam repository. Dengan demikian, proses preprocessing dan training menggunakan dataset yang tersedia pada versi repository yang dijalankan.

Penggunaan DVC membantu menjaga keterlacakan versi dataset yang digunakan dalam eksperimen machine learning.

## 5. Hasil Eksekusi Automation Pipeline

Workflow GitHub Actions berhasil dijalankan setelah perubahan kode dikirim ke branch `main`.

Tahapan pengujian otomatis, preprocessing, training, dan evaluasi dapat dieksekusi. Hasil evaluasi pada salah satu eksekusi pipeline adalah sebagai berikut:

| Metrik | Hasil |
|---|---:|
| RMSE | 1.958586 USD/gram |
| MAE | 1.556379 USD/gram |
| R² | 0.939827 |
| Jumlah data pengujian | 124 |
| Batas maksimum RMSE | 0.95 USD/gram |
| Quality Gate | FAIL |

Nilai RMSE sebesar 1.958586 USD/gram masih melebihi batas maksimum 0.95 USD/gram. Oleh karena itu, model belum memenuhi kriteria quality gate dan registrasi model otomatis dilewati.

Pemeriksaan peningkatan relatif terhadap baseline juga belum dapat dievaluasi karena baseline yang tersedia tidak berasal dari dataset yang sama dan belum memenuhi syarat perbandingan yang valid.

## 6. Evaluasi Hasil Workflow

Status keberhasilan workflow GitHub Actions dan status quality gate merupakan dua hal yang berbeda.

Workflow dapat berhasil menjalankan seluruh tahapan teknis, menghasilkan laporan evaluasi, dan menyelesaikan proses otomatisasi. Namun, hasil evaluasi model tetap dapat menunjukkan status FAIL apabila metrik belum memenuhi kriteria yang ditentukan.

Dengan demikian, keberhasilan eksekusi workflow tidak selalu berarti model berhasil melewati quality gate. Mekanisme ini memastikan bahwa model yang belum memenuhi standar evaluasi tidak otomatis didaftarkan ke MLflow Model Registry.

## 7. Artefak dan Hasil Otomatisasi

Pipeline menghasilkan beberapa artefak penting, antara lain:

- Laporan evaluasi model: `reports/evaluation.json`.
- Workflow otomatisasi: `.github/workflows/mlops-automation.yaml`.
- Script evaluasi dan registrasi model: `src/evaluate_and_register.py`.
- Pengujian otomatis: `tests/test_preprocess.py` dan `tests/test_evaluate.py`.

Laporan evaluasi juga disimpan sebagai workflow artifact dengan nama `model-evaluation-report`, sehingga hasil eksekusi dapat diperiksa kembali melalui halaman GitHub Actions.

## 8. Kesimpulan

Implementasi LK-08 berhasil mengintegrasikan GitHub Actions sebagai pemicu otomatisasi pipeline MLOps. Perubahan kode pada repository dapat memicu proses automated testing, preprocessing, training, dan evaluasi model secara berurutan.

Mekanisme quality gate diterapkan untuk memastikan bahwa registrasi model dilakukan berdasarkan kriteria evaluasi yang telah ditentukan. Pada hasil evaluasi yang diperoleh, model belum memenuhi batas maksimum RMSE sehingga registrasi otomatis tidak dilanjutkan.

Implementasi ini menunjukkan bahwa otomatisasi MLOps tidak hanya bertujuan menjalankan proses machine learning secara otomatis, tetapi juga menerapkan mekanisme kontrol kualitas sebelum model dapat didaftarkan untuk tahap berikutnya.
---

## LK-09 — Docker Compose Orchestration

### 1. Tujuan

Pada tahap LK-09, proyek dikembangkan menggunakan Docker Compose untuk mengorkestrasi seluruh komponen aplikasi MLOps dalam satu lingkungan terintegrasi. Implementasi ini bertujuan menyederhanakan proses deployment, mengatur komunikasi antarlayanan, serta menjaga persistensi data menggunakan Docker volumes.

Arsitektur aplikasi terdiri dari PostgreSQL, MLflow Tracking Server, proses inisialisasi model, FastAPI sebagai layanan inferensi, dan Streamlit sebagai antarmuka pengguna.

### 2. Arsitektur Services

Docker Compose mengelola lima service utama:

| Service | Fungsi | Port |
|---|---|---|
| `postgres` | Menyimpan metadata MLflow menggunakan PostgreSQL | 5432 (internal) |
| `mlflow-server` | Menyediakan MLflow Tracking Server dan Model Registry | 5000 |
| `model-init` | Memigrasikan model lama ke MLflow yang menggunakan PostgreSQL | - |
| `api-service` | Menyediakan REST API untuk inferensi menggunakan FastAPI | 8000 |
| `dashboard` | Menyediakan antarmuka prediksi menggunakan Streamlit | 8501 |

Seluruh service dihubungkan melalui custom bridge network bernama `goldprice-network`.

Komunikasi antarkontainer menggunakan nama service sebagai hostname (service discovery), sehingga setiap service dapat berkomunikasi tanpa perlu mengetahui alamat IP masing-masing.

Contoh komunikasi:
- API mengakses MLflow melalui `http://mlflow-server:5000`.
- Dashboard mengakses API melalui `http://api-service:8000`.
- MLflow mengakses PostgreSQL melalui hostname service `postgres`.

### 3. Docker Network dan Persistent Storage

#### Custom Bridge Network

Docker Compose menggunakan network `goldprice-network` dengan driver `bridge`. Network ini memungkinkan seluruh container berkomunikasi secara internal menggunakan nama service.

#### Named Volumes

Dua named volume digunakan untuk menjaga persistensi data:

| Volume | Mount Point | Fungsi |
|---|---|---|
| `postgres_data` | `/var/lib/postgresql/data` | Menyimpan data PostgreSQL secara persisten |
| `mlflow_artifacts` | `/mlartifacts` | Menyimpan artefak model MLflow secara persisten |

Dengan named volume, data tetap tersedia ketika container dihentikan atau dibuat ulang, selama volume tidak dihapus.

### 4. Konfigurasi Environment

Konfigurasi environment disimpan dalam file `.env` agar informasi koneksi dan kredensial tidak ditulis langsung pada konfigurasi Docker Compose.

Contoh konfigurasi:

```env
POSTGRES_DB=mlflow
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=change_this_password

MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_MODEL_NAME=GoldPriceForecasting
MLFLOW_MODEL_VERSION=1
```

Repository menyediakan file `.env.example` sebagai template konfigurasi. File `.env` lokal dikecualikan dari Git untuk menjaga keamanan kredensial.

### 5. Menjalankan Aplikasi dengan Docker Compose

Pastikan Docker dan Docker Compose tersedia pada environment yang digunakan.

Pada GitHub Codespaces, buat Codespace dari branch `main`, kemudian siapkan konfigurasi environment:

```bash
cp .env.example .env
```

Sesuaikan nilai `POSTGRES_PASSWORD` pada file `.env`.

Setelah konfigurasi selesai, jalankan seluruh service dari root repository menggunakan satu perintah:

```bash
docker compose up -d
```

Docker Compose akan membangun image yang diperlukan dan menjalankan seluruh service sesuai konfigurasi dependensi.

Untuk memeriksa status container:

```bash
docker compose ps -a
```

### 6. Dependency dan Health Check

Konfigurasi `depends_on` digunakan untuk mengatur urutan startup antarlayanan.

Alur dependensi aplikasi:

1. PostgreSQL dijalankan dan menunggu hingga health check menunjukkan status healthy.
2. MLflow Tracking Server dijalankan setelah PostgreSQL sehat.
3. `model-init` dijalankan setelah MLflow sehat untuk memigrasikan model lama ke Model Registry.
4. API dijalankan setelah proses inisialisasi model berhasil diselesaikan.
5. Dashboard dijalankan setelah API berstatus healthy.

Health check diterapkan pada PostgreSQL, MLflow, dan API untuk memastikan layanan siap digunakan.

Service `model-init` merupakan proses sekali jalan (one-shot container). Oleh karena itu, status `Exited (0)` menunjukkan proses migrasi model berhasil diselesaikan, bukan kegagalan container.

### 7. Akses Aplikasi

Setelah seluruh service berjalan, aplikasi dapat diakses melalui alamat berikut:

| Komponen | Alamat |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| FastAPI Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

Pada GitHub Codespaces, akses layanan melalui tab **Ports** dan gunakan alamat forwarding yang disediakan oleh Codespaces.

### 8. Verifikasi Implementasi

#### a. Pemeriksaan Status Container

Perintah:

```bash
docker compose ps -a
```

Hasil implementasi menunjukkan:
- PostgreSQL berstatus `Up (healthy)`.
- MLflow berstatus `Up (healthy)`.
- API berstatus `Up (healthy)`.
- Dashboard berstatus `Up`.
- Model initialization berstatus `Exited (0)`.

#### b. Pengujian Komunikasi API dengan MLflow

Pengujian dilakukan dari dalam container API untuk memastikan komunikasi antarkontainer melalui custom bridge network berjalan dengan baik.

```bash
docker compose exec api-service python -c "import urllib.request; r=urllib.request.urlopen('http://mlflow-server:5000/health'); print('MLflow response:', r.status); print(r.read().decode())"
```

Hasil pengujian:

```text
MLflow response: 200
OK
```

Respons HTTP 200 menunjukkan bahwa API berhasil mengakses MLflow menggunakan nama service `mlflow-server`.

#### c. Pemeriksaan Health API

Perintah:

```bash
curl http://localhost:8000/health
```

Hasil pengujian:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "tracking_uri": "http://mlflow-server:5000"
}
```

Hasil tersebut menunjukkan API berjalan dengan baik dan model berhasil dimuat dari MLflow Model Registry.

#### d. Pengujian Inferensi

Pengujian inferensi dilakukan melalui FastAPI Swagger UI pada endpoint `POST /predict`.

API berhasil memberikan respons HTTP 200 dan menghasilkan prediksi harga emas beserta informasi nama model, versi model, dan satuan prediksi.

Pengujian juga dilakukan melalui Streamlit Dashboard. Hasil prediksi Dashboard sesuai dengan hasil API ketika menggunakan input fitur yang sama.

### 9. Menghentikan Aplikasi

Untuk menghentikan dan menghapus container serta network tanpa menghapus persistent volume, gunakan:

```bash
docker compose down
```

Named volume PostgreSQL dan MLflow tetap dipertahankan sehingga data dapat digunakan kembali ketika aplikasi dijalankan.

Hindari penggunaan `docker compose down -v` apabila data pada persistent volume masih ingin dipertahankan.

### 10. Hasil Implementasi

Implementasi Docker Compose berhasil mengorkestrasi lima service yang saling terintegrasi dalam proyek Gold Price Forecasting.

Hasil pengujian menunjukkan bahwa:
1. PostgreSQL berhasil berjalan sebagai backend penyimpanan metadata MLflow.
2. MLflow Tracking Server berhasil terhubung dengan PostgreSQL.
3. Model berhasil dimigrasikan dan didaftarkan pada MLflow Model Registry dengan nama `GoldPriceForecasting` versi 1 dan status `READY`.
4. FastAPI berhasil memuat model dari MLflow dan menyediakan layanan inferensi.
5. Streamlit Dashboard berhasil terhubung dengan API dan menampilkan hasil prediksi.
6. Komunikasi antarkontainer berhasil dilakukan melalui custom bridge network menggunakan nama service.
7. Named volume berhasil dikonfigurasi untuk menjaga persistensi data PostgreSQL dan artefak MLflow.

Dengan implementasi ini, seluruh komponen aplikasi dapat dijalankan melalui satu konfigurasi Docker Compose. Hal tersebut membuat proses deployment, pengelolaan layanan, dan reproduksi lingkungan aplikasi menjadi lebih terstruktur.