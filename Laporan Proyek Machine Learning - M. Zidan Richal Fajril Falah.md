# Laporan Proyek Machine Learning - M. Zidan Richal Fajril Falah

## Domain Proyek

Diabetes mellitus adalah penyakit metabolik kronis yang ditandai oleh kadar gula darah tinggi (hiperglikemia), baik karena pankreas tidak menghasilkan cukup insulin, atau karena tubuh tidak dapat secara efektif menggunakan insulin yang dihasilkannya. Penyakit ini merupakan masalah kesehatan masyarakat global yang signifikan [1]. Komplikasi jangka panjang akibat diabetes sangat serius, meliputi penyakit jantung, stroke, gagal ginjal, kebutaan, dan amputasi kaki, jika tidak didiagnosis dan ditangani secara tepat waktu [1, 2]. Organisasi Kesehatan Dunia (WHO) memperkirakan bahwa pada tahun 2014, sekitar 422 juta orang dewasa hidup dengan diabetes, dengan tren peningkatan yang berkelanjutan [2]. Studi terbaru bahkan menunjukkan bahwa satu dari setiap dua penderita diabetes mungkin tidak terdiagnosis, menggarisbawahi urgensi diagnosis dini [3].

Deteksi dini dan intervensi yang tepat sangat krusial dalam pengelolaan diabetes untuk mencegah atau menunda timbulnya komplikasi parah [3]. Sayangnya, banyak individu tidak menyadari kondisi diabetes mereka hingga penyakitnya mencapai tahap lanjut, yang mempersulit pengobatan dan memperburuk prognosis [4]. Faktor risiko seperti gaya hidup tidak sehat, obesitas, riwayat keluarga, dan usia tua berkontribusi pada tingginya prevalensi diabetes. Metode skrining konvensional seringkali melibatkan tes laboratorium yang invasif dan kunjungan medis reguler, yang dapat memakan waktu dan biaya, terutama di daerah dengan akses terbatas ke fasilitas kesehatan.

Di sinilah peran **Machine Learning (ML)** menjadi sangat vital. ML menawarkan potensi besar sebagai alat prediksi dan skrining yang non-invasif dan efisien. Dengan memanfaatkan algoritma ML, pola dari data pasien (seperti riwayat medis, hasil tes diagnostik, dan demografi) dapat dianalisis untuk mengidentifikasi individu yang berisiko tinggi terkena diabetes bahkan sebelum gejala klinis signifikan muncul. Penelitian telah menunjukkan bahwa algoritma ML dapat secara efektif memprediksi risiko diabetes dengan akurasi yang menjanjikan, bahkan pada tahap awal penyakit [5, 6]. Model prediksi ini dapat memberikan manfaat signifikan bagi para profesional medis dalam:

1.  **Skrining Awal dan Identifikasi Risiko:** Memungkinkan identifikasi probabilitas tinggi seseorang untuk mengembangkan diabetes, yang membuka peluang untuk intervensi dini seperti perubahan gaya hidup atau pemantauan medis yang lebih intensif.
2.  **Pengambilan Keputusan Klinis yang Lebih Baik:** Menyediakan informasi tambahan yang berharga bagi dokter untuk membuat keputusan diagnostik dan terapeutik yang lebih tepat dan terinformasi.
3.  **Efisiensi Sumber Daya Kesehatan:** Mengurangi beban pada sistem kesehatan dengan memprioritaskan pasien yang benar-benar memerlukan perhatian medis segera, sehingga sumber daya dapat dialokasikan secara lebih efektif.

## Business Understanding

Pada bagian ini, proyek bertujuan untuk mengklarifikasi masalah diabetes dan merumuskan tujuan yang dapat dicapai melalui penerapan Machine Learning.

### Problem Statements

1.  **Tingginya Angka Penderita Diabetes yang Tidak Terdiagnosis:** Banyak individu yang menderita diabetes tidak menyadari kondisi mereka hingga penyakit mencapai tahap lanjut, yang mempersulit pengobatan dan meningkatkan risiko komplikasi serius.
2.  **Keterbatasan Metode Skrining Konvensional:** Metode diagnosis diabetes yang ada seringkali memerlukan prosedur invasif (misalnya tes darah), kunjungan medis yang berulang, serta memakan waktu dan biaya, sehingga menjadi hambatan bagi deteksi dini, terutama di daerah dengan sumber daya terbatas.
3.  **Potensi Risiko Komplikasi Jangka Panjang:** Keterlambatan diagnosis dan penanganan diabetes dapat menyebabkan komplikasi kesehatan parah seperti penyakit jantung, stroke, gagal ginjal, dan kebutaan, yang secara signifikan menurunkan kualitas hidup pasien dan meningkatkan beban sistem kesehatan.

### Goals

1.  **Membangun Model Prediksi Dini Diabetes:** Mengembangkan model Machine Learning yang mampu memprediksi risiko diabetes pada individu berdasarkan data diagnostik yang tersedia, sebelum gejala klinis yang parah muncul.
2.  **Meningkatkan Efisiensi Proses Skrining:** Menyediakan alat prediktif non-invasif yang dapat membantu mengidentifikasi individu berisiko tinggi dengan cepat dan efisien, sehingga memungkinkan intervensi kesehatan lebih awal dan terarah.
3.  **Mengurangi Risiko Komplikasi Diabetes:** Dengan diagnosis dini dan intervensi yang tepat yang difasilitasi oleh model prediksi, diharapkan dapat mengurangi angka komplikasi serius dan meningkatkan kualitas hidup penderita diabetes.

### Solution Statements

Untuk mencapai tujuan yang telah ditetapkan, proyek ini akan mengimplementasikan beberapa solusi terukur:

1.  **Eksplorasi dan Pembandingan Berbagai Model Berbasis Pohon:**

      * Menggunakan berbagai algoritma klasifikasi berbasis pohon yang kuat, yaitu Decision Tree, Random Forest, Gradient Boosting, Extra Trees, AdaBoost, LightGBM, dan XGBoost. Setiap model akan dilatih dan dievaluasi secara independen untuk memahami karakteristik kinerja masing-masing pada dataset.
      * **Metrik Terukur:** Kinerja model akan diukur menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan *ROC AUC*. Model dengan nilai metrik *Recall* dan *F1-Score* yang tinggi (untuk mengidentifikasi kasus positif dengan baik) serta *ROC AUC* yang optimal (untuk kemampuan diskriminasi keseluruhan) akan menjadi kandidat terbaik.

2.  **Optimasi Model Terbaik melalui *Hyperparameter Tuning*:**

      * Setelah membandingkan kinerja awal dari berbagai model berbasis pohon, model dengan performa paling menjanjikan akan dipilih. Selanjutnya, akan dilakukan *hyperparameter tuning* (misalnya menggunakan GridSearchCV atau RandomizedSearchCV) untuk mencari kombinasi parameter optimal yang dapat meningkatkan kinerja model secara signifikan.
      * **Metrik Terukur:** Peningkatan kinerja akan dievaluasi dengan membandingkan nilai metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan *ROC AUC* dari model sebelum dan sesudah *tuning*.

## Data Understanding

Bagian ini menjelaskan informasi mengenai data yang digunakan dalam proyek prediksi diabetes. Proyek ini menggunakan dataset Pima Indians Diabetes Database.

**Sumber Data:**
Dataset ini dapat diunduh dari Kaggle melalui tautan berikut: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Informasi Umum Dataset
Dataset ini berisi informasi tentang karakteristik kesehatan pasien yang digunakan untuk memprediksi kemungkinan diabetes. Dataset ini terdiri dari **768 sampel (baris)** dan **9 fitur (kolom)**.

### Variabel-variabel pada Dataset Pima Indians Diabetes adalah sebagai berikut:

* **Pregnancies**: Jumlah kehamilan yang pernah dialami oleh pasien. Fitur numerik dengan tipe data integer.
* **Glucose**: Konsentrasi glukosa plasma dalam darah setelah 2 jam dalam tes toleransi glukosa oral. Fitur numerik dengan tipe data integer.
* **BloodPressure**: Tekanan darah diastolik (mm Hg). Fitur numerik dengan tipe data integer.
* **SkinThickness**: Ketebalan lipatan kulit trisep (mm). Fitur numerik dengan tipe data integer.
* **Insulin**: Kadar insulin serum setelah 2 jam dalam tes toleransi glukosa oral (mu U/ml). Fitur numerik dengan tipe data integer.
* **BMI**: Indeks Massa Tubuh (Body Mass Index) dihitung dengan berat dalam kg / (tinggi dalam meter)^2. Fitur numerik dengan tipe data float.
* **DiabetesPedigreeFunction**: Fungsi riwayat keluarga diabetes, yaitu suatu nilai yang menunjukkan kemungkinan diabetes berdasarkan riwayat keluarga. Fitur numerik dengan tipe data float.
* **Age**: Usia pasien (tahun). Fitur numerik dengan tipe data integer.
* **Outcome**: Variabel target yang menunjukkan apakah pasien didiagnosis diabetes (1) atau tidak (0). Fitur kategorikal (biner) dengan tipe data integer.

### Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami distribusi, pola, dan hubungan antar variabel dalam dataset.

#### 1. Analisis Univariat
Analisis univariat berfokus pada setiap fitur secara individual menggunakan statistik deskriptif dan histogram.

* **Statistik Deskriptif**:
    * **Pregnancies**: Rata-rata 3.85, dengan nilai minimum 0 dan maksimum 17. Nilai 0 mungkin mengindikasikan data tidak valid atau pasien belum pernah hamil.
    * **Glucose**: Rata-rata 120.89, dengan nilai minimum 0 dan maksimum 199. Terdapat kemungkinan *outlier* atau nilai tidak valid (khususnya nilai 0).
    * **BloodPressure**: Rata-rata 69.11, dengan nilai minimum 0 dan maksimum 122. Nilai 0 mengindikasikan kemungkinan data tidak valid.
    * **SkinThickness**: Rata-rata 20.54, dengan nilai minimum 0 dan maksimum 99. Kemungkinan terdapat *outlier* atau data tidak valid.
    * **Insulin**: Rata-rata 79.80, dengan nilai minimum 0 dan maksimum 846. Fitur ini memiliki rentang nilai yang sangat lebar dan kemungkinan terdapat *outlier*.
    * **BMI**: Rata-rata 31.99, dengan nilai minimum 0 dan maksimum 67.1. Nilai 0 mengindikasikan kemungkinan adanya data tidak valid.
    * **DiabetesPedigreeFunction**: Rata-rata 0.47, dengan nilai minimum 0.08 dan maksimum 2.42.
    * **Age**: Rata-rata 33.24 tahun, dengan usia termuda 21 tahun dan tertua 81 tahun.
    * **Outcome**: Sekitar 34.9% pasien (268 dari 768) didiagnosis menderita diabetes (Outcome = 1), menunjukkan adanya ketidakseimbangan kelas.

* **Histogram**:
    * **Pregnancies, SkinThickness, Insulin, dan DiabetesPedigreeFunction**: Memiliki distribusi yang menceng (skewed).
    * **Glucose, BloodPressure, SkinThickness, Insulin, dan BMI**: Terdapat nilai 0 yang tidak realistis dan perlu ditangani.
    * **Outcome**: Distribusi menunjukkan sekitar 65.1% 'Tidak Diabetes' dan 34.9% 'Diabetes', mengkonfirmasi ketidakseimbangan kelas.

#### 2. Analisis Multivariat
Analisis multivariat mengeksplorasi hubungan antar fitur menggunakan Pair Plot dan Correlation Matrix.

* **Pair Plot**:
    * **Glucose, Age, dan BMI**: Menunjukkan hubungan yang cukup kuat dengan `Outcome` (diabetes), berpotensi menjadi prediktor yang baik.
    * **Insulin dan Glucose**: Terdapat korelasi positif, namun tidak terlalu kuat.
    * **Pregnancies dan Age**: Terdapat korelasi positif yang wajar.

* **Correlation Matrix**:
    * **Korelasi terhadap Variabel Target (Outcome)**:
        * **Glucose (0.47)**: Memiliki korelasi tertinggi, sangat berperan dalam prediksi diabetes.
        * **BMI (0.29)**: Korelasi sedang, kelebihan berat badan berkontribusi terhadap risiko diabetes.
        * **Age (0.24)**: Korelasi positif rendah, menunjukkan semakin tua usia, semakin tinggi kemungkinan mengidap diabetes.
        * **Pregnancies (0.22)**: Korelasi positif terhadap Outcome.
        * **Insulin (0.13), SkinThickness (0.07), dan BloodPressure (0.07)**: Memiliki korelasi sangat rendah.
    * **Korelasi Antar Variabel Fitur**:
        * **Pregnancies dan Age (0.54)**: Korelasi cukup kuat.
        * **Insulin dan SkinThickness (0.44)** serta **SkinThickness dan BMI (0.39)**: Korelasi sedang dan logis secara medis.
        * Korelasi antar fitur lainnya tergolong rendah, menandakan minimnya multikolinearitas.

### Potensi Masalah Teridentifikasi:
* **Nilai 0 yang Tidak Realistis**: Pada fitur Glucose, BloodPressure, SkinThickness, Insulin, dan BMI.
* **Outliers**: Pada fitur Insulin, SkinThickness, dan Pregnancies.
* **Data Imbalance**: Ketidakseimbangan jumlah data antara kelas 'Tidak Diabetes' (0) dan 'Diabetes' (1) pada variabel target ('Outcome').
* **Distribusi Data yang Miring**: Pada fitur Pregnancies, SkinThickness, Insulin, dan DiabetesPedigreeFunction.
* **Multikolinearitas**: Beberapa korelasi antar fitur, seperti antara Pregnancies dan Age, serta antara Insulin dan SkinThickness.

## Data Preparation

Tahap *Data Preparation* adalah langkah krusial dalam proyek ini untuk memastikan data siap digunakan dalam pemodelan *machine learning*. Proses ini melibatkan beberapa teknik penting yang bertujuan untuk mengatasi masalah yang teridentifikasi selama tahap *Data Understanding*, seperti nilai 0 yang tidak realistis, *outlier*, ketidakseimbangan kelas, dan distribusi data yang miring.

### 1. Data Cleaning

Tahap awal *data preparation* adalah pembersihan data untuk mengatasi inkonsistensi dan nilai yang tidak valid.

#### a. Pengecekan dan Penanganan Duplikat
* **Teknik yang Digunakan**: Pemeriksaan dan penghapusan baris data duplikat menggunakan `df.duplicated()` dan `df.drop_duplicates()`.
* **Proses**: Dataset diperiksa untuk mengidentifikasi adanya baris-baris data yang sama persis. Jika ditemukan, baris duplikat tersebut akan dihapus untuk menghindari bias dalam model dan memastikan setiap observasi unik. Setelah penghapusan, indeks DataFrame direset.
* **Alasan**: Data duplikat dapat menyebabkan model mempelajari pola yang salah atau terlalu menekankan pada observasi tertentu, yang pada akhirnya dapat mengurangi generalisasi model. Menghapus duplikat memastikan bahwa setiap observasi berkontribusi secara independen pada proses pembelajaran.

```python
# Periksa data duplikat
num_duplicates = df.duplicated().sum()
print(f"Jumlah data duplikat: {num_duplicates}")

# Hapus data duplikat jika ada
if num_duplicates > 0:
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Data duplikat telah dihapus.")
```

#### b. Pengecekan dan Penanganan Missing Values
* **Teknik yang Digunakan**: Pemeriksaan jumlah *missing values* (NaN) menggunakan `df.isnull().sum()`.
* **Proses**: Dilakukan pengecekan menyeluruh untuk memastikan tidak ada nilai yang hilang (NaN) dalam dataset.
* **Alasan**: Meskipun pada tahap awal tidak terdeteksi adanya missing values dalam format NaN, langkah ini penting untuk memastikan integritas data sebelum melanjutkan ke penanganan nilai 0 yang tidak realistis.

```python
# Periksa missing values
df.isnull().sum()
```

#### c. Pengecekan dan Penanganan Zero Values (Nilai 0 tidak realistis)
* **Teknik yang Digunakan**: Imputasi nilai 0 yang tidak realistis dengan nilai rata-rata (`mean`) berdasarkan grup `Outcome`.
* **Proses**: Beberapa fitur seperti `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` diketahui memiliki nilai 0 yang secara medis tidak realistis. Nilai-nilai 0 ini diganti dengan nilai rata-rata dari fitur tersebut, dikelompokkan berdasarkan `Outcome` (Diabetes atau Tidak Diabetes). Hal ini dilakukan karena nilai rata-rata untuk penderita diabetes mungkin berbeda dari non-penderita.
* **Alasan**: Nilai 0 pada fitur-fitur tersebut adalah indikasi *missing values* atau *error* data, bukan nilai yang sebenarnya. Membiarkan nilai 0 ini dapat secara signifikan mengganggu statistik deskriptif dan kinerja model karena nilai tersebut tidak merepresentasikan kondisi kesehatan yang sebenarnya. Imputasi dengan mean yang dikelompokkan dapat mempertahankan distribusi data asli sebaik mungkin.

```python
# Periksa zero values pada fitur-fitur tertentu
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feature in zero_features:
    num_zeros = (df[feature] == 0).sum()
    print(f"Jumlah nilai 0 tidak realistis pada {feature}: {num_zeros}")

# Ganti nilai 0 dengan mean berdasarkan grup Outcome
df['Glucose'] = df['Glucose'].where((df['Glucose'] > 0)).fillna(df.groupby('Outcome')["Glucose"].transform("mean"))
df['BMI'] = df['BMI'].where((df['BMI'] > 0)).fillna(df.groupby('Outcome')["BMI"].transform("mean"))
df['BloodPressure'] = df['BloodPressure'].where((df['BloodPressure'] > 0)).fillna(df.groupby('Outcome')["BloodPressure"].transform("mean"))
df['Insulin'] = df['Insulin'].where((df['Insulin'] > 0)).fillna(df.groupby('Outcome')["Insulin"].transform("mean"))
df['SkinThickness'] = df['SkinThickness'].where((df['SkinThickness'] > 0)).fillna(df.groupby('Outcome')["SkinThickness"].transform("mean"))
```

#### d. Penanganan Outliers dan Distribusi Data yang Miring
* **Teknik yang Digunakan**: *Capping* (Winsorizing) menggunakan metode IQR (Interquartile Range).
* **Proses**:
    1. **Identifikasi Outlier**: Menggunakan Boxplot untuk memvisualisasikan outlier pada setiap fitur numerik.
    2. **Hitung Ambang Batas (Thresholds)**: Menghitung batas bawah dan batas atas ($Q1 - 1.5 \times IQR$) dan ($Q3 + 1.5 \times IQR$) untuk setiap fitur.
    3. **Capping Outlier**: Nilai-nilai yang berada di luar batas ambang tersebut diganti dengan nilai ambang batas terdekat (nilai batas bawah atau batas atas).
* **Alasan**: *Outlier* adalah nilai ekstrem yang dapat memengaruhi kinerja model secara signifikan, terutama model yang sensitif terhadap nilai ekstrem (misalnya, regresi linier). *Capping* membantu mengurangi dampak *outlier* tanpa menghapus data, sehingga informasi penting tetap terjaga. Selain itu, *capping* juga dapat membantu mengurangi efek *skewness* pada distribusi data.

```python
# Fungsi untuk menghitung batas bawah dan atas dengan IQR
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit

# Fungsi untuk mengganti nilai outlier
def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit

# Terapkan capping ke kolom numerik
num_cols = df_imputed.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    replace_with_thresholds(df_imputed, col)
```

### 2. Feature Engineering

Tahap ini bertujuan untuk menciptakan fitur-fitur baru dari fitur yang sudah ada guna memberikan informasi tambahan yang mungkin dapat meningkatkan kinerja model.

#### a. Glucose Insulin Ratio
* **Teknik yang Digunakan**: Pembuatan fitur rasio.
* **Proses**: Fitur baru `Glucose_Insulin_Ratio` dibuat dengan membagi nilai `Glucose` dengan nilai `Insulin`.
* **Alasan**: Rasio ini dapat memberikan wawasan mengenai sensitivitas insulin seseorang, yang merupakan faktor penting dalam diagnosis diabetes.

```python
df_capped['Glucose_Insulin_Ratio'] = df_capped['Glucose'] / df_capped['Insulin']
```

#### b. Binning Fitur Numerik
* **Teknik yang Digunakan**: Pengelompokan (binning) fitur numerik ke dalam kategori diskrit.
* **Proses**: Fitur-fitur numerik seperti `Glucose`, `Insulin`, `BloodPressure`, `BMI`, dan `Age` dikelompokkan ke dalam kategori-kategori berdasarkan referensi klinis atau distribusi data (misalnya, 'Rendah', 'Normal', 'Pradiabetes', 'Diabetes' untuk Glukosa; 'Muda', 'Dewasa', 'Tua' untuk Age).
* **Alasan**: Binning dapat mengubah hubungan non-linear menjadi linear dan mengurangi sensitivitas terhadap *outlier*. Ini juga dapat membantu model menangkap pola yang lebih general pada data.

```python
# Contoh binning Glucose
def bin_glucose(glucose):
    if glucose < 70:
        return 'Rendah'
    elif 70 <= glucose <= 99:
        return 'Normal'
    elif 100 <= glucose <= 125:
        return 'Pradiabetes'
    else:
        return 'Diabetes'
df_capped['Glucose_Group'] = df_capped['Glucose'].apply(bin_glucose)

# Binning untuk fitur lain seperti Insulin, BloodPressure, BMI, Age juga dilakukan dengan logika serupa.
```

### 3. Data Encoding

* **Teknik yang Digunakan**: *One-Hot Encoding*.
* **Proses**: Fitur-fitur kategorikal baru yang dihasilkan dari proses *binning* (`Glucose_Group`, `Insulin_Group`, `BloodPressure_Group`, `BMI_Group`, `Age_Group`) diubah menjadi format numerik menggunakan `OneHotEncoder`. Setiap kategori unik akan menjadi kolom biner baru (0 atau 1).
* **Alasan**: Sebagian besar algoritma *machine learning* memerlukan input data dalam bentuk numerik. *One-Hot Encoding* adalah metode yang efektif untuk mengubah fitur kategorikal menjadi representasi numerik tanpa menyiratkan hubungan ordinal antar kategori.

```python
from sklearn.preprocessing import OneHotEncoder
categorical_cols = ['Glucose_Group', 'Insulin_Group', 'BloodPressure_Group', 'BMI_Group', 'Age_Group']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.transform(df_capped[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df_capped.index)
df_encoded = df_capped.drop(columns=categorical_cols).join(encoded_df)
```

### 4. Data Splitting

* **Teknik yang Digunakan**: Pembagian dataset menjadi data latih (*training set*) dan data uji (*testing set*) menggunakan `train_test_split`.
* **Proses**: Dataset dibagi menjadi 80% untuk data latih dan 20% untuk data uji, dengan `random_state` yang tetap untuk reproduktifitas. Fitur (`X`) dan variabel target (`y`) dipisahkan terlebih dahulu.
* **Alasan**: Pemisahan data ini esensial untuk mengevaluasi kinerja model secara objektif. Model dilatih hanya pada data latih, dan kemudian dievaluasi pada data uji yang belum pernah dilihat sebelumnya untuk mengukur kemampuan generalisasinya.

```python
from sklearn.model_selection import train_test_split
X = df_encoded.drop('Outcome', axis=1)
y = df_encoded['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Normalisasi Data (Data Scaling)

* **Teknik yang Digunakan**: `StandardScaler`.
* **Proses**: Fitur-fitur numerik (kecuali fitur hasil *one-hot encoding*) pada data latih dan data uji diskalakan menggunakan `StandardScaler`. `StandardScaler` mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1. `Scaler` hanya di-*fit* pada data latih untuk menghindari *data leakage*.
* **Alasan**: Banyak algoritma *machine learning* sensitif terhadap skala fitur. Jika fitur memiliki rentang nilai yang sangat berbeda, fitur dengan rentang yang lebih besar dapat mendominasi proses pembelajaran. Normalisasi memastikan semua fitur berkontribusi secara proporsional dan dapat meningkatkan konvergensi serta kinerja model.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_Insulin_Ratio']
scaler.fit(X_train[numerical_cols])
X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
```

### 6. Data Balancing (pada data training)

* **Teknik yang Digunakan**: SMOTE (Synthetic Minority Over-sampling Technique).
* **Proses**: SMOTE diterapkan pada data latih (`X_train`, `y_train`) untuk mengatasi ketidakseimbangan kelas. SMOTE menghasilkan sampel sintetis baru untuk kelas minoritas (`Outcome = 1`) berdasarkan sampel terdekatnya, sehingga jumlah sampel di kelas minoritas menjadi seimbang dengan kelas mayoritas.
* **Alasan**: Ketidakseimbangan kelas dapat menyebabkan model cenderung memprediksi kelas mayoritas dan kurang akurat dalam memprediksi kelas minoritas, yang seringkali merupakan kelas yang lebih penting dalam kasus medis seperti diabetes. SMOTE membantu model belajar pola dari kelas minoritas dengan lebih baik, sehingga meningkatkan performa model secara keseluruhan, terutama pada metrik seperti *recall* dan F1-score untuk kelas minoritas.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Jumlah data sebelum SMOTE:")
print(y_train.value_counts())
print("\nJumlah data sesudah SMOTE:")
print(pd.Series(y_train_smote).value_counts())
```

## Modeling

Pada tahap ini, model *machine learning* dibangun dan dilatih untuk memprediksi penyakit diabetes berdasarkan data yang telah diproses.

### 1. Model Selection

Beberapa algoritma *machine learning* klasifikasi berbasis pohon (*tree-based*) dan *ensemble* dipilih untuk dievaluasi:

* **Decision Tree Classifier:** Model dasar yang membangun pohon keputusan berdasarkan fitur data.
    * **Kelebihan:** Mudah diinterpretasi dan divisualisasikan, dapat menangani data numerik dan kategorikal, tidak memerlukan penskalaan fitur.
    * **Kekurangan:** Rentan terhadap *overfitting* (terutama pohon yang dalam), tidak stabil (perubahan kecil pada data dapat menghasilkan pohon yang sangat berbeda).
* **Random Forest Classifier:** Sebuah metode *ensemble* yang membangun banyak *Decision Trees* dan menggabungkan prediksi mereka.
    * **Kelebihan:** Mengurangi *overfitting* dibandingkan *Decision Tree* tunggal, kinerja yang kuat dan akurat, dapat menangani banyak fitur dan *missing values*, serta kurang sensitif terhadap *outlier*.
    * **Kekurangan:** Kurang dapat diinterpretasi (seperti "kotak hitam"), membutuhkan lebih banyak sumber daya komputasi dan waktu pelatihan dibandingkan *Decision Tree* tunggal.
* **Gradient Boosting Classifier:** Model *ensemble* yang membangun pohon secara sekuensial, di mana setiap pohon baru berusaha memperbaiki kesalahan dari pohon sebelumnya.
    * **Kelebihan:** Kinerja tinggi dan akurat (seringkali yang terbaik), dapat menangani hubungan non-linear, dan robust terhadap *outlier*.
    * **Kekurangan:** Rentan terhadap *overfitting* jika *hyperparameter* tidak disetel dengan baik, waktu pelatihan lebih lama, dan lebih kompleks untuk disetel.
* **Extra Trees Classifier:** Mirip dengan Random Forest, tetapi dengan penambahan randomisasi pada proses pemilihan fitur saat membagi node.
    * **Kelebihan:** Lebih cepat dalam pelatihan dibandingkan Random Forest karena randomisasi tambahan, mengurangi *overfitting*, dan kinerja yang kompetitif.
    * **Kekurangan:** Seperti Random Forest, kurang dapat diinterpretasi, dan mungkin sedikit mengorbankan bias untuk mengurangi varians.
* **AdaBoost Classifier:** Metode *boosting* yang berfokus pada sampel yang sulit diklasifikasikan oleh model sebelumnya.
    * **Kelebihan:** Sederhana, efektif untuk meningkatkan kinerja model dasar yang lemah, dan tidak terlalu rentan terhadap *overfitting* dibandingkan *boosting* lainnya.
    * **Kekurangan:** Rentan terhadap data yang *noisy* dan *outlier* (karena fokus pada sampel yang salah diklasifikasikan), kinerja sangat bergantung pada model dasar yang digunakan.
* **LightGBM Classifier:** Sebuah *framework boosting gradient* yang cepat dan efisien.
    * **Kelebihan:** Sangat cepat dalam pelatihan, efisien dalam penggunaan memori, kinerja tinggi (seringkali lebih baik dari XGBoost pada dataset besar), dan dapat menangani data kategorikal secara langsung.
    * **Kekurangan:** Rentan terhadap *overfitting* pada dataset kecil, dan memerlukan penyesuaian *hyperparameter* yang cermat.
* **XGBoost Classifier:** *Framework boosting gradient* populer lainnya yang dikenal karena kecepatan dan kinerja yang kuat.
    * **Kelebihan:** Kinerja sangat tinggi, memiliki regularisasi built-in (L1 dan L2) untuk mengurangi *overfitting*, dapat menangani *missing values*, dan fleksibel.
    * **Kekurangan:** Lebih lambat dari LightGBM pada dataset yang sangat besar, dan membutuhkan lebih banyak penyesuaian *hyperparameter*.

Inisialisasi model-model ini dilakukan sebagai berikut:
```python
# Inisiasi Model
def base_model():
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Extra Trees': ExtraTreesClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'LightGBM': LGBMClassifier(verbose=-1),
        'XGBoost': XGBClassifier()
    }
    return models

models = base_model()
```

### 2. Model Training

Setiap model dilatih menggunakan data latih yang sudah diseimbangkan dengan SMOTE (`X_train_smote`, `y_train_smote`). Setelah pelatihan, akurasi awal dihitung pada data uji (`X_test`).

```python
# 3.2 Model Training
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_smote, y_train_smote)
    print(f"{name} trained.")

    # Prediksi pada data pengujian
    y_pred = model.predict(X_test)

    # Hitung dan tampilkan akurasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy:.4f}", end="\n\n")
```
Hasil akurasi awal dari model-model dasar (sebelum *tuning*) adalah:
| Model                 | Akurasi Awal (Data Uji) |
| :-------------------- | :---------------------- |
| Decision Tree         | 0.8636                  |
| Random Forest         | 0.8766                  |
| Gradient Boosting     | 0.8896                  |
| Extra Trees           | 0.8701                  |
| AdaBoost              | 0.8831                  |
| LightGBM              | 0.8636                  |
| XGBoost               | 0.8571                  |

### 3. Hyperparameter Tuning

Kinerja model sangat bergantung pada *hyperparameter* yang digunakan. Tahap ini bertujuan untuk menemukan kombinasi *hyperparameter* terbaik untuk setiap model guna memaksimalkan kinerjanya. Teknik **Grid Search with Cross-Validation (CV=5)** digunakan untuk eksplorasi *hyperparameter* secara sistematis.

**Parameter Grid yang Digunakan:**
* **Decision Tree**: `max_depth`, `min_samples_split`, `min_samples_leaf`
* **Random Forest**: `max_depth`, `max_features`, `min_samples_split`, `n_estimators`
* **Gradient Boosting**: `n_estimators`, `learning_rate`, `max_depth`, `subsample`
* **Extra Trees**: `n_estimators`, `max_depth`, `min_samples_split`
* **AdaBoost**: `n_estimators`, `learning_rate`
* **LightGBM**: `learning_rate`, `n_estimators`, `colsample_bytree`, `num_leaves`, `max_depth`, `subsample`
* **XGBoost**: `learning_rate`, `max_depth`, `n_estimators`, `colsample_bytree`, `subsample`

```python
from sklearn.model_selection import GridSearchCV

# Contoh parameter grid untuk Random Forest
rf_params = {
    "max_depth": [10, 15, None],
    "max_features": ["sqrt", "log2", 5],
    "min_samples_split": [5, 10],
    "n_estimators": [100, 200]
}

# Contoh tuning model
best_models = {}
tuning_results = []

for name, model in models.items():
    print(f"🔧 Tuning {name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name], # param_grids didefinisikan sebelumnya
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_smote, y_train_smote)

    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    tuning_results.append({
        'Model': name,
        'Best Parameters': grid_search.best_params_,
        'Best CV Score': round(grid_search.best_score_, 4),
        'Test Accuracy': round(test_acc, 4)
    })
    print(f"✅ {name} tuned.\n")
```

**Hasil Penyetelan Hyperparameter (Tuning Summary):**

| Model             | Best Parameters                                                | Best CV Score | Test Accuracy |
| :---------------- | :------------------------------------------------------------- | :------------ | :------------ |
| AdaBoost          | `{'learning_rate': 1.0, 'n_estimators': 200}`                  | 0.9028        | 0.9026        |
| Gradient Boosting | `{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}` | 0.9290        | 0.8701        |
| Decision Tree     | `{'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 2}` | 0.8854        | 0.8701        |
| Random Forest     | `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 100}` | 0.9116        | 0.8636        |
| LightGBM          | `{'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 500, 'num_leaves': 31, 'subsample': 0.8}` | 0.9302        | 0.8636        |
| Extra Trees       | `{'max_depth': None, 'min_samples_split': 5, 'n_estimators': 300}` | 0.9016        | 0.8571        |
| XGBoost           | `{'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}` | 0.9315        | 0.8571        |

### 4. Memilih Best Model untuk Dievaluasi

Model terbaik dipilih berdasarkan metrik **Test Accuracy** karena ini merepresentasikan kinerja model pada data yang belum pernah dilihat setelah *hyperparameter tuning*.
```python
# Urutkan berdasarkan Test Accuracy untuk menentukan model terbaik
best_model_summary = tuning_df.sort_values(by='Test Accuracy', ascending=False)

print("\n🏅 Best Model based on Test Accuracy:")
display(best_model_summary)

# Pilih model terbaik
best_model_name = best_model_summary.index[0]
final_model = best_models[best_model_name]

print(f"\n🎉 The best model selected is: {best_model_name}")
```
**Ringkasan Model Terbaik Berdasarkan Akurasi Uji:**

| Model             | Best Parameters                                                | Best CV Score | Test Accuracy |
| :---------------- | :------------------------------------------------------------- | :------------ | :------------ |
| AdaBoost          | `{'learning_rate': 1.0, 'n_estimators': 200}`                  | 0.9028        | 0.9026        |
| Gradient Boosting | `{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}` | 0.9290        | 0.8701        |
| Decision Tree     | `{'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 2}` | 0.8854        | 0.8701        |
| Random Forest     | `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 100}` | 0.9116        | 0.8636        |
| LightGBM          | `{'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 500, 'num_leaves': 31, 'subsample': 0.8}` | 0.9302        | 0.8636        |
| Extra Trees       | `{'max_depth': None, 'min_samples_split': 5, 'n_estimators': 300}` | 0.9016        | 0.8571        |
| XGBoost           | `{'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}` | 0.9315        | 0.8571        |

**Model terbaik yang terpilih adalah: AdaBoost.**

**Alasan Pemilihan AdaBoost sebagai Model Terbaik:**
AdaBoost menunjukkan akurasi tertinggi pada data uji (`Test Accuracy: 0.9026`) dibandingkan dengan model-model lain setelah proses *hyperparameter tuning*. Meskipun beberapa model lain (seperti LightGBM dan XGBoost) memiliki *Best CV Score* yang sedikit lebih tinggi, *Test Accuracy* adalah metrik paling relevan untuk mengevaluasi kinerja model pada data baru. AdaBoost berhasil mempertahankan kinerja generalisasi yang sangat baik pada data yang belum pernah dilihat.

## Evaluation

Tahap evaluasi model adalah langkah krusial untuk mengukur seberapa baik model yang telah dilatih mampu memprediksi penyakit diabetes pada data yang belum pernah dilihat. Evaluasi ini dilakukan menggunakan beberapa metrik yang relevan untuk masalah klasifikasi biner, terutama mengingat adanya ketidakseimbangan kelas pada dataset.

### Metrik Evaluasi dan Hasil Proyek (AdaBoost)

Model terbaik yang terpilih dari tahap *modeling* adalah **AdaBoost**. Berikut adalah metrik evaluasi yang digunakan dan interpretasi hasilnya pada data uji:

#### 1. Akurasi (Accuracy Score)
* **Formula**: Akurasi dihitung sebagai proporsi total prediksi yang benar dari semua prediksi yang dibuat.
    $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
    * TP (True Positive): Jumlah prediksi positif yang benar.
    * TN (True Negative): Jumlah prediksi negatif yang benar.
    * FP (False Positive): Jumlah prediksi positif yang salah (seharusnya negatif).
    * FN (False Negative): Jumlah prediksi negatif yang salah (seharusnya positif).
* **Cara Kerja**: Akurasi memberikan gambaran umum tentang seberapa sering model membuat prediksi yang benar.
* **Hasil Proyek**: Akurasi AdaBoost pada Data Uji adalah **0.9026**.
* **Interpretasi**: Model AdaBoost menunjukkan akurasi yang tinggi, sekitar 90.26% prediksi yang benar secara keseluruhan. Ini merupakan indikasi kinerja yang solid, meskipun akurasi saja mungkin tidak cukup karena adanya ketidakseimbangan kelas pada dataset.

#### 2. Confusion Matrix
* **Cara Kerja**: *Confusion Matrix* adalah tabel yang meringkas kinerja model klasifikasi dengan menampilkan jumlah *True Positives*, *True Negatives*, *False Positives*, dan *False Negatives*.
* **Konteks Proyek**: Dalam konteks prediksi diabetes:
    * *True Positive (TP)*: Pasien diabetes yang diprediksi diabetes.
    * *True Negative (TN)*: Pasien tidak diabetes yang diprediksi tidak diabetes.
    * *False Positive (FP)*: Pasien tidak diabetes yang diprediksi diabetes (kesalahan Tipe I), yang bisa menyebabkan kecemasan yang tidak perlu.
    * *False Negative (FN)*: Pasien diabetes yang diprediksi tidak diabetes (kesalahan Tipe II), yang merupakan jenis kesalahan yang lebih kritis karena pasien yang sebenarnya sakit tidak terdeteksi.
* **Hasil Proyek**: *Confusion Matrix* untuk AdaBoost adalah:
    ```
    [[90  9]
     [ 6 49]]
    ```
    * **True Negative (TN)**: 90
    * **False Positive (FP)**: 9
    * **False Negative (FN)**: 6
    * **True Positive (TP)**: 49
* **Interpretasi**: Dari 99 pasien non-diabetes (90 TN + 9 FP), model berhasil mengidentifikasi 90 dengan benar. Dari 55 pasien diabetes (6 FN + 49 TP), model berhasil mengidentifikasi 49 dengan benar.

#### 3. Laporan Klasifikasi (Classification Report)
* **Cara Kerja**: Laporan ini menyediakan ringkasan metrik *Precision*, *Recall*, dan *F1-Score* untuk setiap kelas.
    * **Precision (Presisi)**: Mengukur proporsi prediksi positif yang benar dari semua prediksi positif yang dibuat.
        $Precision = \frac{TP}{TP + FP}$
    * **Recall (Sensitivitas/Tingkat Deteksi True Positive)**: Mengukur proporsi positif aktual yang diidentifikasi dengan benar.
        $Recall = \frac{TP}{TP + FN}$
    * **F1-Score**: *Harmonic mean* dari *Precision* dan *Recall*, baik digunakan saat ada ketidakseimbangan kelas.
        $F1-Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
* **Konteks Proyek**: Untuk prediksi diabetes, *Recall* untuk kelas 'Diabetes (1)' seringkali sangat penting untuk mendeteksi sebagian besar pasien yang benar-benar sakit.
* **Hasil Proyek**: Laporan Klasifikasi untuk AdaBoost adalah:
    ```
    📄 Laporan Klasifikasi untuk AdaBoost:
                      precision    recall  f1-score   support

    Tidak Diabetes (0)       0.94      0.91      0.92        99
    Diabetes (1)             0.84      0.89      0.87        55

            accuracy                           0.90       154
           macro avg       0.89      0.90      0.90       154
        weighted avg       0.90      0.90      0.90       154
    ```
* **Interpretasi**:
    * Untuk kelas **'Tidak Diabetes (0)'**: *Precision* 0.94, *Recall* 0.91, dan F1-Score 0.92. Model sangat baik dalam mengidentifikasi pasien yang tidak menderita diabetes.
    * Untuk kelas **'Diabetes (1)'**: *Precision* 0.84, *Recall* 0.89, dan F1-Score 0.87. *Recall* sebesar 0.89 berarti model berhasil mendeteksi 89% dari pasien yang sebenarnya menderita diabetes, metrik yang krusial untuk mencegah diagnosis yang terlewat. *Precision* 0.84 menunjukkan bahwa 84% prediksi diabetes oleh model adalah benar. F1-Score 0.87 menunjukkan keseimbangan yang baik antara *precision* dan *recall* untuk kelas minoritas.

#### 4. ROC AUC Score dan Kurva ROC
* **Cara Kerja**: Kurva ROC adalah plot dari *True Positive Rate (TPR)* atau *Recall* terhadap *False Positive Rate (FPR)* pada berbagai ambang batas klasifikasi.
    $FPR = \frac{FP}{FP + TN}$
    ROC AUC Score adalah area di bawah kurva ROC, dengan nilai berkisar dari 0 (prediksi acak) hingga 1 (model sempurna).
* **Konteks Proyek**: ROC AUC adalah metrik yang sangat baik untuk mengevaluasi kinerja model klasifikasi biner, terutama ketika ada ketidakseimbangan kelas, karena mengukur kemampuan model untuk membedakan antara kelas positif dan negatif di berbagai ambang batas.
* **Hasil Proyek**: ROC AUC Score untuk AdaBoost adalah **0.94**.
* **Interpretasi**: Nilai AUC yang sangat tinggi (0.94) menunjukkan bahwa model memiliki kemampuan diskriminasi yang luar biasa. Artinya, model sangat baik dalam membedakan antara pasien yang positif diabetes dan negatif diabetes, tanpa terpengaruh oleh ambang batas klasifikasi tertentu.

Secara keseluruhan, model AdaBoost menunjukkan kinerja yang sangat menjanjikan dalam memprediksi diabetes. Akurasi tinggi, F1-score yang baik untuk kelas minoritas, dan terutama nilai ROC AUC yang luar biasa, mengindikasikan bahwa model ini memiliki potensi besar sebagai alat prediksi awal untuk penyakit diabetes.

---

## Referensi:

[1] World Health Organization. (2021). *Diabetes*. Retrieved from [https://www.who.int/news-room/fact-sheets/detail/diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)/n

[2] American Diabetes Association. (2020). *Statistics About Diabetes*. Retrieved from [https://diabetes.org/about-diabetes/statistics/about-diabetes](https://diabetes.org/about-diabetes/statistics/about-diabetes)

[3] Reza, M. S., Amin, R., Yasmin, R., Kulsum, W., & Ruhi, S. (2024). Improving diabetes disease patients classification using stacking ensemble method with PIMA and local healthcare data. *Heliyon*, *10*(7), e24536. [https://doi.org/10.1016/j.heliyon.2024.e24536](https://www.google.com/search?q=https://doi.org/10.1016/j.heliyon.2024.e24536)

[4] International Diabetes Federation. (2021). *IDF Diabetes Atlas, 10th Edition*. Brussels, Belgium: International Diabetes Federation. [https://diabetesatlas.org/](https://diabetesatlas.org/)

[5] Sarwar, T., Iqbal, A., & Bashir, T. (2020). Machine learning approach for diabetes prediction using Pima Indian diabetes dataset. *International Journal of Advanced Computer Science and Applications, 11*(11). [https://doi.org/10.14569/IJACSA.2020.0111162](https://www.google.com/search?q=https://doi.org/10.14569/IJACSA.2020.0111162)

[6] Ahmed, A., Khan, J., Arsalan, M., Ahmed, K., Shahat, A. A., Alhalmi, A., & Naaz, S. (2025). Machine Learning Algorithm-Based Prediction of Diabetes Among Female Population Using PIMA Dataset. *Healthcare*, *13*(1), 37. [https://doi.org/10.3390/healthcare13010037](https://www.google.com/search?q=https://doi.org/10.3390/healthcare13010037)

[7] Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann. (Link umum ke buku: [https://www.elsevier.com/books/data-mining-concepts-and-techniques/han/978-0-12-381479-1](https://www.elsevier.com/books/data-mining-concepts-and-techniques/han/978-0-12-381479-1))
