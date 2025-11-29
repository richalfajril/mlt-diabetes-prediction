<div align="center">

# 🩺 MLT1-DIABETES-PREDICTION

_Early Detection of Diabetes Risk using Tree-Based Models_

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

- [Overview](#overview)
- [Dataset](#dataset)
- [Metodologi dan Pemodelan](#metodologi-dan-pemodelan)
- [Hasil dan Evaluasi](#hasil-dan-evaluasi)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Struktur Proyek](#struktur-proyek)

---

## Overview

Proyek ini merupakan implementasi **Machine Learning Terapan 1 (MLT)** yang bertujuan untuk membangun model prediksi dini risiko **Diabetes Mellitus** menggunakan dataset diagnostik **Pima Indians Diabetes Database** dari Kaggle.

Diabetes merupakan penyakit metabolik kronis yang sering tidak terdiagnosis sampai komplikasi serius muncul. Proyek ini mengembangkan sistem skrining non-invasif berbasis machine learning untuk mengidentifikasi individu berisiko tinggi secara dini, memungkinkan intervensi kesehatan yang lebih cepat dan efektif.

### Fokus Utama Proyek:

1. **Eksplorasi dan Perbandingan Model**: Evaluasi tujuh model klasifikasi berbasis pohon (_tree-based models_) termasuk Decision Tree, Random Forest, Gradient Boosting, Extra Trees, AdaBoost, LightGBM, dan XGBoost.
2. **Data Preprocessing & Feature Engineering**: Penanganan nilai 0 yang tidak realistis, penanganan outlier dengan capping (IQR), serta pembuatan fitur turunan (`Glucose_Insulin_Ratio`).
3. **Penyeimbangan Data**: Menggunakan **SMOTE** (Synthetic Minority Over-sampling Technique) untuk mengatasi ketidakseimbangan kelas pada data training.
4. **Optimasi Model**: Hyperparameter tuning menggunakan Grid Search with Cross-Validation untuk memaksimalkan performa model.

### Model Terbaik:

Model **Random Forest** dengan akurasi **88.96%** pada data uji terpilih sebagai model final setelah proses optimasi, menunjukkan kemampuan prediksi yang sangat baik untuk early detection diabetes.

## Dataset

Proyek ini menggunakan dataset **Pima Indians Diabetes Database** yang tersedia di Kaggle.

| Fitur                      | Deskripsi                                     |
| :------------------------- | :-------------------------------------------- |
| `Pregnancies`              | Jumlah kehamilan.                             |
| `Glucose`                  | Konsentrasi glukosa plasma (2 jam tes).       |
| `BloodPressure`            | Tekanan darah diastolik (mm Hg).              |
| `SkinThickness`            | Ketebalan lipatan kulit trisep (mm).          |
| `Insulin`                  | Kadar insulin serum (2 jam tes).              |
| `BMI`                      | Indeks Massa Tubuh (Body Mass Index).         |
| `DiabetesPedigreeFunction` | Riwayat keluarga diabetes.                    |
| `Age`                      | Usia pasien (tahun).                          |
| **`Outcome`**              | **Target:** 1 = Diabetes, 0 = Tidak Diabetes. |

**Catatan Khusus:** Data memiliki masalah **ketidakseimbangan kelas** dan **nilai 0 yang tidak realistis secara medis** pada fitur seperti `Glucose`, `BloodPressure`, `BMI`, dll., yang memerlukan pra-pemrosesan khusus.

---

## Metodologi dan Pemodelan

## Metodologi dan Pemodelan

### Data Preprocessing & Feature Engineering

1. **Penanganan Duplikat Data**: Pengecekan dan penghapusan baris data yang sama persis untuk memastikan setiap observasi unik.

2. **Pengecekan Missing Values**: Verifikasi bahwa tidak ada nilai NaN dalam dataset.

3. **Penanganan Nilai 0 yang Tidak Realistis**:

   - Fitur-fitur seperti `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` memiliki nilai 0 yang secara medis tidak realistis.
   - Nilai-nilai 0 tersebut diimputasi dengan **nilai rata-rata (mean)** yang dikelompokkan berdasarkan `Outcome` (diabetes vs non-diabetes) untuk mempertahankan distribusi data asli.

4. **Penanganan Outlier**:

   - Menggunakan teknik **Capping (Winsorizing)** dengan metode **IQR (Interquartile Range)**.
   - Nilai-nilai ekstrem yang berada di luar batas ambang diganti dengan nilai batas terdekat.
   - Ini membantu mengurangi dampak outlier tanpa menghilangkan data.

5. **Feature Engineering**:

   - Membuat fitur turunan: `Glucose_Insulin_Ratio = Glucose / Insulin`
   - Fitur ini memberikan wawasan tentang sensitivitas insulin seseorang, faktor penting dalam diagnosis diabetes.

6. **Data Splitting**:

   - Dataset dibagi menjadi 80% training dan 20% testing dengan `random_state=42` untuk reproduktifitas.
   - `X = df.drop('Outcome', axis=1)` menghasilkan 9 fitur.
   - `y = df['Outcome']` sebagai target.

7. **Normalisasi Data (Data Scaling)**:

   - Menggunakan **StandardScaler** untuk menskalakan data ke mean=0 dan std=1.
   - Hanya 7 fitur yang diskala: `['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_Insulin_Ratio']`
   - Fitur `BloodPressure` dan `SkinThickness` tidak diskala (sesuai proses training original).
   - Scaler hanya di-fit pada data training untuk menghindari data leakage.

8. **Data Balancing (SMOTE)**:
   - Diterapkan pada data training untuk mengatasi ketidakseimbangan kelas (65.1% non-diabetes, 34.9% diabetes).
   - **SMOTE** (Synthetic Minority Over-sampling) membuat sampel sintetis untuk kelas minoritas.
   - Menghasilkan jumlah sampel yang seimbang antara kedua kelas pada data training.

### Pemodelan dan Optimasi

**Model-Model yang Dievaluasi:**

| Model                 | Deskripsi                                                                                                    |
| :-------------------- | :----------------------------------------------------------------------------------------------------------- |
| **Decision Tree**     | Model dasar yang membangun pohon keputusan berdasarkan fitur data.                                           |
| **Random Forest**     | Ensemble method yang menggabungkan banyak decision trees untuk mengurangi overfitting.                       |
| **Gradient Boosting** | Ensemble method yang membangun pohon secara sekuensial, setiap pohon memperbaiki kesalahan pohon sebelumnya. |
| **Extra Trees**       | Mirip Random Forest dengan penambahan randomisasi pada pemilihan fitur.                                      |
| **AdaBoost**          | Boosting method yang fokus pada sampel yang sulit diklasifikasikan.                                          |
| **LightGBM**          | Framework boosting gradient yang cepat dan efisien.                                                          |
| **XGBoost**           | Framework boosting gradient populer dengan performa tinggi.                                                  |

**Hyperparameter Tuning:**

- Menggunakan **Grid Search with Cross-Validation (CV=5)** untuk menemukan kombinasi hyperparameter optimal.
- Setiap model dikembangankan dengan parameter grid spesifik untuk mengeksplorasi space hyperparameter secara sistematis.
- Hasil tuning menunjukkan peningkatan signifikan dari akurasi awal baseline models.

---

## Hasil dan Evaluasi

### Akurasi Model Baseline (Sebelum Hyperparameter Tuning)

| Model             | Akurasi |
| :---------------- | :------ |
| Decision Tree     | 0.8442  |
| Random Forest     | 0.8701  |
| Gradient Boosting | 0.8766  |
| Extra Trees       | 0.8766  |
| AdaBoost          | 0.8701  |
| LightGBM          | 0.8766  |
| XGBoost           | 0.8636  |

### Hasil Hyperparameter Tuning (Ranked by Test Accuracy)

| Model             | Best CV Score | Test Accuracy | Best Parameters                                                                                                             |
| :---------------- | :------------ | :------------ | :-------------------------------------------------------------------------------------------------------------------------- |
| **Random Forest** | **0.9190**    | **0.8896**    | `{'max_depth': 10, 'max_features': 5, 'min_samples_split': 5, 'n_estimators': 200}`                                         |
| Extra Trees       | 0.9178        | 0.8896        | `{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 300}`                                                            |
| AdaBoost          | 0.8978        | 0.8831        | `{'learning_rate': 1.0, 'n_estimators': 200}`                                                                               |
| XGBoost           | 0.9240        | 0.8701        | `{'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}`                    |
| Decision Tree     | 0.8853        | 0.8766        | `{'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 15}`                                                          |
| Gradient Boosting | 0.9339        | 0.8636        | `{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}`                                             |
| LightGBM          | 0.9302        | 0.8636        | `{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 500, 'num_leaves': 31, 'subsample': 0.8}` |

### Model Terbaik: Random Forest

**Random Forest** dipilih sebagai model final berdasarkan **Test Accuracy tertinggi (0.8896 / 88.96%)** pada data yang belum pernah dilihat setelah hyperparameter tuning.

#### Kinerja Detail (Random Forest pada Data Uji)

| Metrik                  | Nilai           |
| :---------------------- | :-------------- |
| **Accuracy**            | 0.8896 (88.96%) |
| **Precision (Kelas 0)** | 0.93            |
| **Recall (Kelas 0)**    | 0.94            |
| **F1-Score (Kelas 0)**  | 0.94            |
| **Precision (Kelas 1)** | 0.83            |
| **Recall (Kelas 1)**    | 0.82            |
| **F1-Score (Kelas 1)**  | 0.83            |
| **ROC AUC Score**       | ~0.94           |

#### Confusion Matrix (Random Forest)

```
                 Predicted Negative  Predicted Positive
Actual Negative        94                     5
Actual Positive         10                    45
```

**Interpretasi:**

- Model berhasil mengidentifikasi **94 dari 99** pasien non-diabetes (94% True Negative Rate).
- Model berhasil mengidentifikasi **45 dari 55** pasien diabetes (82% True Positive Rate / Recall).
- Dari perspektif medis, Recall 82% untuk diabetes sangat penting karena meminimalkan False Negative (pasien diabetes yang terlewat).

#### Feature Importance (Top 10)

| Rank | Feature                  | Importance |
| :--- | :----------------------- | :--------- |
| 1    | Insulin                  | 0.4488     |
| 2    | SkinThickness            | 0.1126     |
| 3    | Age                      | 0.1087     |
| 4    | Glucose                  | 0.1022     |
| 5    | Glucose_Insulin_Ratio    | 0.0913     |
| 6    | BMI                      | 0.0456     |
| 7    | DiabetesPedigreeFunction | 0.0403     |
| 8    | BloodPressure            | 0.0269     |
| 9    | Pregnancies              | 0.0237     |

**Key Insights:**

- **Insulin** adalah faktor paling dominan (44.88%) - berhubungan dengan resistensi insulin, penyebab utama diabetes tipe 2.
- **SkinThickness & BMI** - indikator lemak tubuh dan risiko metabolik.
- **Age** - semakin tua, semakin tinggi risiko diabetes.
- **Glucose_Insulin_Ratio** - fitur rekayasa yang terbukti berkontribusi signifikan (9.13%).
- **DiabetesPedigreeFunction** - faktor genetika tetap berpengaruh.

---

## Getting Started

Untuk memulai dan menjalankan proyek ini di mesin lokal Anda, ikuti petunjuk di bawah ini.

### Prerequisites

Anda memerlukan lingkungan Python yang dikonfigurasi. Pastikan Anda memiliki:

- **Python 3.9+**
- **Git**

### Installation

1. **Kloning repositori:**

   ```bash
   git clone https://github.com/richalfajril/mlt-diabetes-prediction.git
   cd mlt-diabetes-prediction
   ```

2. **Membuat Virtual Environment (Recommended):**

   ```bash
   # Menggunakan venv
   python3 -m venv venv
   source venv/bin/activate  # Untuk macOS/Linux
   # atau
   venv\Scripts\activate  # Untuk Windows
   ```

3. **Instal dependensi:**

   ```bash
   pip install -r requirements.txt
   ```

   Jika file `requirements.txt` tidak ada, instal secara manual:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm joblib streamlit kagglehub
   ```

### Usage

Proyek ini menyediakan dua cara untuk menggunakan hasil pemodelan:

#### 1. **Jupyter Notebook (Analisis Lengkap)**

Untuk melihat seluruh proses dari EDA, preprocessing, modeling, hingga evaluasi:

```bash
jupyter notebook MLT1_Diabetes_Prediction.ipynb
```

Buka file di browser dan jalankan setiap sel secara berurutan untuk mereplikasi seluruh workflow proyek.

#### 2. **Streamlit Web Application (Prediksi Interaktif)**

Untuk menggunakan model dalam aplikasi web yang user-friendly:

```bash
streamlit run app.py
```

Aplikasi akan membuka di `http://localhost:8501` dengan interface untuk:

- Input data kesehatan pasien (usia, BMI, kadar glukosa, insulin, dll)
- Analisis risiko diabetes secara real-time
- Visualisasi probabilitas prediksi
- Interpretasi hasil dengan interface yang intuitif

#### 3. **Python Script (Training & Prediction)**

Untuk menjalankan logika pemodelan dalam format script Python:

```bash
python mlt1_diabetes_prediction.py
```

---

## Struktur Proyek

Repositori ini diorganisir sebagai berikut:

```
mlt-diabetes-prediction/
├── MLT1_Diabetes_Prediction.ipynb          # Jupyter Notebook utama dengan analisis lengkap
├── app.py                                   # Aplikasi Streamlit untuk prediksi interaktif
├── mlt1_diabetes_prediction.py              # Script Python berisi logika pemodelan
├── final_diabetes_model.joblib              # Model Random Forest terbaik (serialized)
├── scaler.joblib                            # StandardScaler untuk preprocessing (serialized)
├── requirements.txt                         # Daftar dependencies Python
├── README.md                                # Dokumentasi proyek (file ini)
└── Laporan Proyek Machine Learning - M. Zidan Richal Fajril Falah.md  # Laporan lengkap

```

### Deskripsi File

| File                                                                | Deskripsi                                                                                                                                                     |
| :------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MLT1_Diabetes_Prediction.ipynb`                                    | Notebook Jupyter utama yang berisi seluruh pipeline: EDA, data preprocessing, feature engineering, model training, hyperparameter tuning, dan evaluasi model. |
| `app.py`                                                            | Aplikasi web Streamlit yang menyediakan interface user-friendly untuk memprediksi risiko diabetes secara real-time.                                           |
| `mlt1_diabetes_prediction.py`                                       | Versi script Python dari seluruh pipeline pemodelan (hasil konversi dari notebook).                                                                           |
| `final_diabetes_model.joblib`                                       | Model Random Forest yang telah dilatih dan dioptimasi, disimpan dalam format joblib untuk digunakan kembali.                                                  |
| `scaler.joblib`                                                     | StandardScaler yang telah di-fit pada data training, digunakan untuk preprocessing data input dalam prediksi.                                                 |
| `requirements.txt`                                                  | File yang mendaftar semua library Python dan versinya yang diperlukan untuk menjalankan proyek.                                                               |
| `Laporan Proyek Machine Learning - M. Zidan Richal Fajril Falah.md` | Laporan komprehensif yang menjelaskan secara detail domain masalah, business understanding, methodology, dan hasil evaluasi.                                  |

---

## Model Artifacts

Model dan scaler yang telah dilatih disimpan sebagai artifacts untuk memungkinkan reusability:

- **`final_diabetes_model.joblib`**: Random Forest model dengan hyperparameter optimal yang siap digunakan untuk prediksi baru.
- **`scaler.joblib`**: StandardScaler yang telah di-fit, memastikan data input baru di-scale dengan cara yang sama seperti training data.

Artifacts ini dimuat oleh `app.py` menggunakan `joblib.load()` untuk memberikan prediksi yang konsisten dan akurat.

---

## Technical Stack

- **Python 3.9+**: Bahasa pemrograman utama
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Ensemble Methods**: Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM
- **Data Balancing**: Imbalanced-learn (SMOTE)
- **Web Application**: Streamlit
- **Model Serialization**: Joblib
- **Data Source**: Kaggle (Pima Indians Diabetes Database)

---

## Key Findings & Insights

### Preprocessing Highlights

✅ **Imputasi Nilai 0**: Mengganti nilai 0 yang tidak realistis dengan mean berdasarkan outcome grup
✅ **Outlier Handling**: IQR-based capping mengurangi dampak nilai ekstrem tanpa kehilangan data
✅ **Feature Engineering**: Glucose_Insulin_Ratio terbukti kontribusi signifikan (9.13%)
✅ **SMOTE Balancing**: Mengatasi imbalance dari 65:35 menjadi seimbang pada training data

### Model Performance

✅ **Akurasi**: 88.96% - Prediksi yang akurat secara keseluruhan
✅ **Recall (Diabetes)**: 82% - Mampu mendeteksi 82% dari kasus diabetes aktual
✅ **Precision (Diabetes)**: 83% - Prediksi positif yang reliable
✅ **Feature Importance**: Insulin dominan (44.88%), diikuti SkinThickness, Age, dan Glucose

### Clinical Implications

🩺 Model dapat menjadi alat screening awal yang efektif
🩺 Recall 82% penting untuk deteksi dini dan mencegah False Negatives
🩺 Feature importance align dengan pengetahuan medis (insulin resistance → diabetes)

---

## Author

**M. Zidan Richal Fajril Falah**

Proyek ini adalah hasil pembelajaran dalam kursus Machine Learning Terapan 1 (MLT1).

---

## License

Proyek ini dilisensikan di bawah MIT License. Lihat file LICENSE untuk detail lebih lanjut.

---

## Referensi

1. WHO - Diabetes Fact Sheets: https://www.who.int/news-room/fact-sheets/detail/diabetes
2. Kaggle - Pima Indians Diabetes Database: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
3. Scikit-learn Documentation: https://scikit-learn.org/
4. Imbalanced-learn (SMOTE): https://imbalanced-learn.org/
5. XGBoost & LightGBM Documentation
6. Streamlit Documentation: https://streamlit.io/
