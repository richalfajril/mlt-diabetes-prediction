<div align="center">

# 🩺 MLT1-DIABETES-PREDICTION

*Early Detection of Diabetes Risk using Tree-Based Models*

[![Last Commit](https://img.shields.io/github/last-commit/richalfajril/mlt-diabetes-prediction)](https://github.com/richalfajril/mlt-diabetes-prediction/commits/main)
![Jupyter Notebook](https://img.shields.io/badge/jupyter%20notebook-100.0%25-orange)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Built with the tools and technologies:

![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1D96CE?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4A9D0A?style=for-the-badge&logo=lightgbm&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-368DE4?style=for-the-badge&logo=matplotlib&logoColor=white)

</div>

---

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Metodologi dan Pemodelan](#metodologi-dan-pemodelan)
* [Hasil dan Evaluasi](#hasil-dan-evaluasi)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Usage](#usage)
* [Struktur Proyek](#struktur-proyek)

---

## Overview

Proyek ini merupakan implementasi **Machine Learning Terapan** yang bertujuan untuk membangun model prediksi dini risiko **Diabetes** menggunakan dataset diagnostik Pima Indians.

Fokus utama proyek ini adalah **eksplorasi dan optimasi model klasifikasi berbasis pohon** (*Tree-based models*) seperti **AdaBoost**, **XGBoost**, dan **Random Forest**. Proses ini mencakup penanganan nilai hilang (imputasi), penanganan *outlier*, *feature engineering* untuk meningkatkan informasi, dan penyeimbangan data menggunakan **SMOTE**.

Model terbaik yang dioptimasi diharapkan dapat menjadi alat skrining non-invasif yang efisien untuk mengidentifikasi individu berisiko tinggi.

## Dataset

Proyek ini menggunakan dataset **Pima Indians Diabetes Database** yang tersedia di Kaggle.

| Fitur | Deskripsi |
| :--- | :--- |
| `Pregnancies` | Jumlah kehamilan. |
| `Glucose` | Konsentrasi glukosa plasma (2 jam tes). |
| `BloodPressure` | Tekanan darah diastolik (mm Hg). |
| `SkinThickness`| Ketebalan lipatan kulit trisep (mm). |
| `Insulin` | Kadar insulin serum (2 jam tes). |
| `BMI` | Indeks Massa Tubuh (Body Mass Index). |
| `DiabetesPedigreeFunction`| Riwayat keluarga diabetes. |
| `Age` | Usia pasien (tahun). |
| **`Outcome`** | **Target:** 1 = Diabetes, 0 = Tidak Diabetes. |

**Catatan Khusus:** Data memiliki masalah **ketidakseimbangan kelas** dan **nilai 0 yang tidak realistis secara medis** pada fitur seperti `Glucose`, `BloodPressure`, `BMI`, dll., yang memerlukan pra-pemrosesan khusus.

---

## Metodologi dan Pemodelan

### Data Preprocessing & Feature Engineering

1.  **Penanganan Nilai 0:** Nilai 0 pada fitur-fitur klinis diimputasi dengan **nilai rata-rata (`mean`)** yang dikelompokkan berdasarkan `Outcome`.
2.  **Penanganan *Outlier***: Digunakan teknik **Capping (Winsorizing)** menggunakan metode IQR untuk mengurangi dampak nilai ekstrem.
3.  ***Feature Engineering***: Dibuat fitur-fitur baru (`Glucose_Insulin_Ratio`) dan dilakukan **Binning/Grouping** untuk fitur-fitur numerik (`Age`, `BMI`, `Glucose`) untuk mengekstrak informasi kategorikal.
4.  **Penyeimbangan Data:** Data latih diseimbangkan menggunakan **SMOTE** (Synthetic Minority Over-sampling Technique).
5.  **Scaling:** Data dinormalisasi menggunakan **StandardScaler**.

### Pemodelan dan Optimasi

Proyek ini membandingkan **tujuh (7)** model klasifikasi berbasis pohon, termasuk:
* **AdaBoost Classifier**
* **XGBoost Classifier**
* **Random Forest Classifier**
* **Gradient Boosting Classifier**
* **LightGBM Classifier**
* **Extra Trees Classifier**
* **Decision Tree Classifier**

Semua model menjalani proses **Hyperparameter Tuning** menggunakan **Grid Search with Cross-Validation (CV=5)** untuk menemukan kombinasi parameter terbaik.

---

## Hasil dan Evaluasi

Model **AdaBoost Classifier** terpilih sebagai model dengan kinerja optimal setelah proses *Hyperparameter Tuning*.

### Kinerja Model Terbaik (AdaBoost Classifier)

Model dievaluasi dengan fokus pada metrik **Recall** untuk meminimalkan *False Negative* (pasien diabetes yang tidak terdeteksi).

| Metrik | Nilai | Interpretasi |
| :--- | :--- | :--- |
| **Akurasi Uji** | **0.9026** (90.26%) | Tingkat prediksi benar secara keseluruhan. |
| **Recall (Kelas 1)** | **0.89** | 89% dari semua kasus diabetes yang sebenarnya terdeteksi oleh model. |
| **F1-Score (Kelas 1)** | 0.87 | Keseimbangan yang baik antara Precision dan Recall untuk kelas diabetes. |
| **ROC AUC Score** | **0.94** | Kemampuan diskriminasi model yang sangat baik. |

**Kesimpulan:** Model AdaBoost memberikan hasil yang sangat menjanjikan dengan akurasi dan **Recall** tinggi, menjadikannya model yang efektif untuk skrining dini diabetes.

---

## Getting Started

Untuk memulai dan menjalankan proyek ini di mesin lokal Anda, ikuti petunjuk di bawah ini.

### Prerequisites

Anda memerlukan lingkungan Python yang dikonfigurasi. Pastikan Anda memiliki:
* **Python 3.9+**
* **Git**

### Installation

1.  **Kloning repositori:**
    ```bash
    git clone [https://github.com/richalfajril/mlt-diabetes-prediction.git](https://github.com/richalfajril/mlt-diabetes-prediction.git)
    cd mlt-diabetes-prediction
    ```

2.  **Instal dependensi:**
    Proyek ini membutuhkan pustaka populer seperti `pandas`, `scikit-learn`, `imblearn` (untuk SMOTE), `xgboost`, dan `lightgbm`.
    ```bash
    # Jika menggunakan requirements.txt (asumsi ada)
    # pip install -r requirements.txt
    
    # Jika tidak ada requirements.txt, instal secara manual:
    pip install numpy pandas matplotlib seaborn scikit-learn imblearn xgboost lightgbm jupyter
    ```
    *Disarankan untuk membuat dan mengaktifkan lingkungan virtual sebelum instalasi.*

### Usage

Proyek ini dikembangkan dalam format Jupyter Notebook, yang berisi semua langkah dari EDA, pra-pemrosesan, pemodelan, hingga evaluasi.

1.  **Jalankan Jupyter Notebook:**
    ```bash
    jupyter notebook MLT1_Diabetes_Prediction.ipynb
    ```
2.  **Eksekusi:** Buka file `MLT1_Diabetes_Prediction.ipynb` di *browser* Anda dan jalankan setiap sel secara berurutan untuk mereplikasi seluruh alur kerja proyek.

---

## Struktur Proyek

Repositori ini diorganisir sebagai berikut:

* `MLT1_Diabetes_Prediction.ipynb`: **Notebook Jupyter utama** yang berisi seluruh kode analisis data, pra-pemrosesan, *feature engineering*, *model training*, *hyperparameter tuning*, dan evaluasi.
* `Laporan Proyek Machine Learning - M. Zidan Richal Fajril Falah.md`: Laporan komprehensif proyek dalam format Markdown.
* `README.md`: Berkas dokumentasi utama proyek.
* `mlt1_diabetes_prediction.py`: (Asumsi) Versi script Python dari logika pemodelan.
* `requirements.txt`: (Asumsi) Daftar semua pustaka yang diperlukan untuk proyek.
