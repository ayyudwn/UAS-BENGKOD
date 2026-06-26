# UAS Bengkel Koding Data Science

## Identitas Mahasiswa

| Keterangan | Isi |
|---|---|
| Nama | Suryani Ayu Dewanti|
| NIM | A11.2023.15018 |
| Kelompok | DS 01 |
| Mata Kuliah | Bengkel Koding Data Science |
| Project | Customer Churn Prediction |

---

## Judul Project

**Customer Churn Prediction Menggunakan Machine Learning pada Sales and Marketing Customer Dataset**

---

## Deskripsi Project

Project ini merupakan tugas Ujian Akhir Semester Bengkel Koding Data Science. Project ini berfokus pada prediksi **customer churn**, yaitu kondisi ketika pelanggan berhenti menggunakan layanan atau tidak lagi melakukan aktivitas pembelian.

Dataset yang digunakan adalah **Sales and Marketing Customer Dataset**. Dataset ini berisi informasi pelanggan seperti data demografis, aktivitas penggunaan layanan, riwayat transaksi, interaksi pelanggan, serta informasi pemasaran.

Target prediksi pada project ini adalah kolom:

- `churn = 0` berarti pelanggan tidak churn
- `churn = 1` berarti pelanggan churn

Tujuan utama project ini adalah membangun model machine learning untuk memprediksi kemungkinan pelanggan churn, mengevaluasi beberapa model, memilih model terbaik, dan melakukan deployment menggunakan Streamlit Cloud.

---

## Bidang Project

Project ini berada pada bidang:

**Data Science / Machine Learning untuk Bisnis dan Pemasaran**

Secara lebih spesifik, project ini termasuk ke dalam:

- Customer Churn Prediction
- Supervised Learning
- Classification
- Sales and Marketing Analytics
- Customer Relationship Management Analytics

---

## Dataset

Dataset yang digunakan:

**Sales and Marketing Dataset**

Sumber dataset:

https://www.kaggle.com/datasets/bhaskerpaul/sales-and-marketing-dataset

Jumlah data berdasarkan soal:

- 15.000 records
- 30 kolom

Beberapa fitur yang digunakan dalam dataset antara lain:

- `customer_id`
- `gender`
- `age`
- `country`
- `city`
- `signup_date`
- `last_purchase_date`
- `acquisition_channel`
- `device_type`
- `subscription_type`
- `is_premium_user`
- `total_visits`
- `avg_session_time`
- `pages_per_session`
- `email_open_rate`
- `email_click_rate`
- `total_spent`
- `avg_order_value`
- `discount_used`
- `coupon_code`
- `support_tickets`
- `refund_requested`
- `delivery_delay_days`
- `payment_method`
- `satisfaction_score`
- `nps_score`
- `marketing_spend_per_user`
- `lifetime_value`
- `last_3_month_purchase_freq`
- `churn`

---

## Tujuan Project

Tujuan dari project ini adalah:

1. Melakukan Exploratory Data Analysis atau EDA untuk memahami karakteristik dataset.
2. Membangun model prediksi churn menggunakan tiga kategori model machine learning.
3. Membandingkan performa model pada beberapa skenario eksperimen.
4. Melakukan hyperparameter tuning untuk meningkatkan performa model.
5. Memilih model terbaik berdasarkan hasil evaluasi.
6. Melakukan deployment model terbaik menggunakan Streamlit Cloud.

---

## Skenario Modeling

Project ini akan menghasilkan total **9 model**, yang berasal dari kombinasi:

- 3 kategori model
- 3 skenario eksperimen

### Kategori Model

1. Model konvensional  
   Contoh: Logistic Regression atau KNN

2. Ensemble Bagging  
   Contoh: Random Forest

3. Ensemble Voting  
   Contoh: VotingClassifier yang menggabungkan beberapa model konvensional

### Skenario Eksperimen

1. Direct Modeling  
   Model dilatih langsung tanpa preprocessing dan tanpa hyperparameter tuning.

2. Modeling dengan Preprocessing  
   Model dilatih setelah data melalui proses preprocessing.

3. Hyperparameter Tuning  
   Model dioptimasi menggunakan metode tuning seperti GridSearchCV atau RandomizedSearchCV.

---

## Evaluasi Model

Metrik evaluasi yang digunakan dalam project ini adalah:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Deployment

Model terbaik akan disimpan dalam format:

- `.pkl`
- atau `.joblib`

Kemudian model akan digunakan pada aplikasi Streamlit yang berisi:

- Pemuatan model
- Form input fitur pelanggan
- Proses prediksi churn
- Tampilan hasil prediksi
- Penjelasan fitur
- Visualisasi pendukung jika diperlukan

Aplikasi akan diuji secara lokal terlebih dahulu, kemudian diunggah ke GitHub dan dideploy ke Streamlit Cloud.

---

## Struktur Project Sementara

```bash
UAS-BENGKOD/
│
├── notebook/
│   └── A11_2023_15018_UAS_BENGKOD.ipynb
│
├── dataset/
│   └── sales_and_marketing.csv
│
├── model/
│   └── best_model.pkl
│
├── app.py
├── requirements.txt
└── README.md