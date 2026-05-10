# 🌫️ UCI Air Quality Analysis
> End-to-end Data Science project — Data Cleaning, EDA, dan Machine Learning pada dataset kualitas udara dari UCI ML Repository

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Tentang Project

Project ini merupakan analisis end-to-end pada **UCI Air Quality Dataset** — dataset sensor kualitas udara per jam dari sebuah kota di Italia (Maret 2004 – Februari 2005).

Motivasi utama: *"Bisakah kita memprediksi kadar polutan berbahaya hanya dari data sensor murah dan informasi cuaca?"*

Dataset ini menarik karena punya dua jenis pengukuran:
- **Sensor murah** (PT08.Sx) yang dipasang di lapangan
- **Alat referensi laboratorium** sebagai ground truth

Sehingga kita bisa mengukur seberapa akurat sensor murah tersebut, dan membangun model yang bisa mengkalibrasinya.

---

## 📊 Dataset

| Info | Detail |
|---|---|
| Sumber | [UCI ML Repository — Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) |
| Periode | Maret 2004 – Februari 2005 |
| Frekuensi | Per jam |
| Total data | ~9.358 baris |
| Fitur | 15 kolom (sensor + cuaca + waktu) |
| Polutan utama | CO, Benzene, NOx, NO2 |

---

## 🗂️ Struktur Project

```
uci-air-quality-analysis/
│
├── 📓 air_quality_complete.ipynb   ← Notebook utama (semua part)
│
├── 📁 data/
│   ├── AirQualityUCI.csv           ← Data mentah (download otomatis)
│   ├── air_quality_cleaned.csv     ← Output Part 1 (setelah cleaning)
│   └── air_quality_clustered.csv  ← Output Part 3 (dengan label cluster)
│
└── 📄 README.md
```

---

## 🔬 Alur Analisis

### Part 1 — Data Cleaning & Preprocessing
- Handling missing values yang dikodekan sebagai **-200**
- Parsing DateTime dari format non-standar (`DD/MM/YYYY HH.MM.SS`)
- Drop kolom dengan missing >90% (`NMHC_ref`)
- Imputasi dengan **time-series interpolation** (lebih tepat dari mean)
- Deteksi dan penanganan outlier dengan **IQR Winsorization**
- Feature engineering: `hour`, `dayofweek`, `is_weekend`, `is_rush_hour`, `season`

### Part 2 — Exploratory Data Analysis (EDA)
- Distribusi tiap polutan (histogram + KDE)
- Tren time series dengan rolling mean 7 hari
- **Rush hour analysis**: jam berapa udara paling berbahaya?
- Perbandingan kualitas udara: weekday vs weekend
- Heatmap korelasi antar variabel
- Scatter CO vs Benzene: validasi korelasi emisi kendaraan
- **Accuracy check**: sensor murah vs alat referensi

### Part 3 — Machine Learning
**Regresi** (prediksi kadar CO):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.44 | 0.66 | 0.74 |
| Random Forest | 0.44 | 0.66 | 0.74 |
| **XGBoost** ⭐ | 0.45 | 0.67 | 0.73 |

> *Nilai akan terisi setelah notebook dijalankan*

**Clustering** (K-Means):
- Elbow method + Silhouette Score untuk menentukan K optimal
- Visualisasi cluster dengan **PCA 2D**
- Interpretasi cluster: identifikasi pola "jam sibuk berpolusi", "malam bersih", dll.

---

## 💡 Key Insights

- 🕐 **Jam paling berpolusi** terjadi pada pagi hari (rush hour) dan sore hari, konsisten dengan pola lalu lintas kendaraan
- 📅 **Weekend lebih bersih** dibanding hari kerja — CO rata-rata lebih rendah secara signifikan
- 🔗 **CO dan Benzene** berkorelasi sangat tinggi karena keduanya berasal dari pembakaran bahan bakar yang tidak sempurna
- 🌡️ **Suhu berkorelasi negatif** dengan beberapa polutan — udara panas membantu dispersi polutan
- 🔧 **Sensor murah** memiliki korelasi tinggi dengan alat referensi, namun ada drift yang perlu kalibrasi

---

## 🛠️ Tech Stack

| Library | Kegunaan |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `matplotlib` & `seaborn` | Visualisasi |
| `missingno` | Visualisasi missing values |
| `scikit-learn` | Machine learning & preprocessing |
| `xgboost` | Gradient boosting |

---

## 🚀 Cara Menjalankan

**Opsi 1 — Google Colab (Rekomendasi)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload `air_quality_complete.ipynb` ke Google Colab
2. Jalankan semua cell dari atas ke bawah (`Runtime → Run all`)
3. Dataset akan otomatis terdownload dari UCI Repository

**Opsi 2 — Lokal**

```bash
# Clone repo
git clone https://github.com/username/uci-air-quality-analysis.git
cd uci-air-quality-analysis

# Install dependencies
pip install numpy pandas matplotlib seaborn missingno scikit-learn xgboost

# Jalankan notebook
jupyter notebook air_quality_complete.ipynb
```

---

## 📁 Output yang Dihasilkan

| File | Deskripsi |
|---|---|
| `air_quality_cleaned.csv` | Dataset bersih siap analisis |
| `air_quality_clustered.csv` | Dataset dengan label hasil clustering |

---

## 👤 Author

**[Nama Kamu]**
Mahasiswa Teknik Informatika | Data & AI Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/username)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/username)

---

## 📄 Referensi

- S. De Vito et al., *On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario*, Sensors and Actuators B: Chemical, 2008
- [UCI ML Repository — Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
