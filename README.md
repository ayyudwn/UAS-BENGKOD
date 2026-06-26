# Customer Churn Prediction

Project UAS Bengkel Koding Data Science untuk memprediksi apakah pelanggan berpotensi berhenti menggunakan layanan (*churn*) atau tetap menggunakan layanan.

- **Nama:** Suryani Ayu Dewanti
- **NIM:** A11.2023.15018
- **Target:** `churn`
- **Kelas 0:** tidak churn
- **Kelas 1:** churn

## Gambaran Project

Project ini membangun dan membandingkan tiga kategori model *machine learning* pada tiga skenario eksperimen. Total terdapat sembilan hasil model yang dibandingkan menggunakan metrik klasifikasi.

### Model yang digunakan

1. Logistic Regression sebagai model konvensional.
2. Random Forest sebagai model *ensemble bagging*.
3. Voting Classifier sebagai gabungan beberapa model.

### Skenario eksperimen

1. Direct Modeling + SMOTE.
2. Preprocessing + SMOTE.
3. Hyperparameter Tuning + SMOTE.

Model terbaik dipilih dari seluruh sembilan hasil eksperimen berdasarkan F1-score, recall, dan accuracy. F1-score menjadi pertimbangan utama karena distribusi target mengalami ketidakseimbangan kelas.

## Dataset

Dataset yang digunakan adalah **Sales and Marketing Customer Dataset** dari Kaggle.

Sumber dataset:

```text
https://www.kaggle.com/datasets/bhaskerpaul/sales-and-marketing-dataset
```

Dataset memuat informasi demografis pelanggan, aktivitas penggunaan layanan, riwayat transaksi, interaksi pemasaran, tingkat kepuasan, dan status churn.

## Alur Pengerjaan

### 1. Exploratory Data Analysis

Tahapan EDA meliputi:

- Pemeriksaan struktur dan tipe data.
- Statistik deskriptif.
- Pemeriksaan dan visualisasi missing value.
- Visualisasi distribusi target churn.
- Visualisasi outlier.
- Encoding fitur kategorikal sebelum analisis korelasi.
- Heatmap korelasi fitur numerik kontinu dan kategori hasil encoding.
- Analisis hubungan setiap fitur terhadap target churn.

### 2. Feature Engineering

Beberapa fitur diubah agar lebih sesuai untuk model:

- `coupon_code` diubah menjadi `has_coupon_code`.
- `signup_date` dan `last_purchase_date` diubah menjadi:
  - `customer_tenure_days`
  - `days_since_last_purchase`

### 3. Feature Selection

Tiga fitur dengan pengaruh terendah dihapus:

- `customer_id`
- `discount_used`
- `avg_order_value`

`customer_id` dihapus karena hanya berfungsi sebagai identitas pelanggan. Dua fitur lainnya dihapus berdasarkan hasil analisis pengaruh terhadap target.

### 4. Penanganan Missing Value

Missing value ditangani sesuai tipe dan distribusi fitur:

- Fitur numerik diisi menggunakan mean atau median.
- Median digunakan pada fitur dengan distribusi miring atau memiliki nilai ekstrem.
- Mean digunakan pada fitur dengan distribusi relatif simetris.
- Fitur kategorikal diisi menggunakan modus atau nilai yang paling sering muncul.
- Nilai kosong pada `coupon_code` tidak langsung diimputasi, tetapi diubah menjadi informasi ada atau tidaknya kupon.

Pada aplikasi Streamlit, pilihan seperti `age tidak diketahui` akan diubah menjadi `NaN`, kemudian ditangani oleh imputer yang sama seperti saat training.

### 5. Penanganan Outlier

Outlier divisualisasikan pada tahap EDA dan ditangani menggunakan **IQR Capping**.

Nilai di luar batas bawah dan batas atas tidak dihapus, tetapi dibatasi pada nilai IQR yang telah dihitung dari data train.

### 6. Encoding dan Scaling

- Fitur kategorikal diubah menggunakan `OneHotEncoder`.
- Fitur numerik di-*scaling* menggunakan `StandardScaler`.
- Seluruh transformer disusun menggunakan `ColumnTransformer` dan `Pipeline`.

### 7. Train, Validation, dan Test Split

Data dibagi menjadi data train, validation, dan test.

SMOTE hanya diterapkan pada:

- Data train.
- Data validation.

Data test tetap menggunakan distribusi asli agar evaluasi akhir menggambarkan kondisi nyata.

### 8. Penanganan Imbalance

Ketidakseimbangan kelas ditangani menggunakan **SMOTE**.

SMOTE membuat sampel sintetis pada kelas minoritas agar jumlah kelas lebih seimbang. SMOTE tidak diterapkan pada data test dan tidak dijalankan saat pengguna melakukan prediksi pada aplikasi Streamlit.

### 9. Evaluasi Model

Model dievaluasi menggunakan:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Hasil dari seluruh sembilan eksperimen dibandingkan untuk menentukan model terbaik.

### 10. Hyperparameter Tuning

Metode tuning yang digunakan:

- `GridSearchCV` untuk Logistic Regression.
- `RandomizedSearchCV` untuk Random Forest.
- `RandomizedSearchCV` untuk Voting Classifier.

Tuning dilakukan untuk memperoleh kombinasi parameter terbaik dari masing-masing model.

## Deployment

Model terbaik beserta seluruh komponen preprocessing disimpan dalam:

```text
customer_churn_deployment.joblib
```

File tersebut memuat:

- Model terbaik.
- Nama dan skenario model.
- Preprocessor.
- Batas IQR.
- Daftar fitur input.
- Daftar fitur setelah preprocessing.
- Tanggal referensi.
- Daftar fitur yang dihapus.
- Strategi imputasi.
- Informasi SMOTE.
- Hasil evaluasi model.

Aplikasi Streamlit memanggil file `.joblib`, bukan menjalankan notebook secara langsung.

Alur prediksi pada aplikasi:

```text
Input pengguna
→ Feature engineering
→ IQR capping
→ Imputasi missing value
→ OneHotEncoder
→ StandardScaler
→ Prediksi model
→ Hasil churn atau tidak churn
```

## Fitur Aplikasi Streamlit

Aplikasi menyediakan:

- Form input dengan nama kolom yang mengikuti dataset.
- Pilihan untuk memasukkan nilai tidak diketahui sebagai missing value.
- Prediksi churn atau tidak churn.
- Probabilitas churn.
- Informasi model dan metrik evaluasi.
- Penjelasan preprocessing.
- Informasi penerapan SMOTE.
- Panduan pengisian fitur.
- Unduh hasil prediksi dalam format CSV.

## Struktur Project

```text
UAS-BENGKOD/
├── app.py
├── customer_churn_deployment.joblib
├── requirements.txt
├── README.md
└── A11_2023_15018_UAS_BENGKOD_FINAL_REVISI.ipynb
```

Keterangan:

- `app.py`: aplikasi Streamlit.
- `customer_churn_deployment.joblib`: model dan komponen preprocessing.
- `requirements.txt`: daftar library yang diperlukan.
- `README.md`: dokumentasi project.
- File `.ipynb`: proses EDA, preprocessing, modeling, tuning, evaluasi, dan penyimpanan model.

## Instalasi dan Menjalankan Project

### 1. Clone repository

```bash
git clone https://github.com/ayyudwn/UAS-BENGKOD.git
cd UAS-BENGKOD
```

### 2. Membuat virtual environment

```bash
python -m venv venv
```

### 3. Mengaktifkan virtual environment

Untuk Windows CMD:

```bash
venv\Scripts\activate
```

Untuk Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Untuk Linux atau macOS:

```bash
source venv/bin/activate
```

### 4. Menginstal dependency

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Menjalankan aplikasi

```bash
python -m streamlit run app.py
```

Aplikasi biasanya dapat diakses melalui:

```text
http://localhost:8501
```

### 6. Menghentikan aplikasi

Tekan:

```text
Ctrl + C
```

Untuk keluar dari virtual environment:

```bash
deactivate
```

## Requirements

Isi minimal `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
joblib
```

Versi `scikit-learn` dan `imbalanced-learn` sebaiknya disamakan dengan versi yang digunakan saat membuat file `.joblib` agar tidak terjadi masalah kompatibilitas.

## Deployment ke Streamlit Community Cloud

1. Pastikan `app.py`, `customer_churn_deployment.joblib`, `requirements.txt`, dan `README.md` sudah di-*push* ke GitHub.
2. Buka Streamlit Community Cloud.
3. Hubungkan akun GitHub.
4. Pilih repository `UAS-BENGKOD`.
5. Pilih branch utama.
6. Isi main file path dengan:

```text
app.py
```

7. Jalankan deployment.
8. Pastikan file `.joblib` dapat dimuat dan seluruh form prediksi berjalan.

## Catatan

- SMOTE hanya digunakan pada proses training dan validation.
- Data test tidak melalui SMOTE.
- Input pengguna pada Streamlit juga tidak melalui SMOTE.
- Notebook digunakan untuk membangun model.
- Aplikasi hanya memanggil file `customer_churn_deployment.joblib`.
- Model tetap perlu dievaluasi secara berkala apabila digunakan pada data baru.

## Lisensi

Project ini dibuat untuk keperluan akademik UAS Bengkel Koding Data Science.
