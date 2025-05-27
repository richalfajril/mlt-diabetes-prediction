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

Proyek ini menggunakan dataset **Pima Indians Diabetes Database** yang tersedia di Kaggle: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Dataset ini merupakan kumpulan catatan medis diagnostik yang spesifik untuk wanita keturunan Pima Indian, berusia minimal 21 tahun. Tujuan utama dari dataset ini adalah untuk memprediksi apakah seorang pasien menderita diabetes (variabel `Outcome`) berdasarkan beberapa pengukuran diagnostik.

Dataset ini terdiri dari 768 entri dengan 9 kolom (8 fitur prediktor dan 1 variabel target).

### Variabel-variabel pada Pima Indians Diabetes Database adalah sebagai berikut:

  - `Pregnancies`: Jumlah kehamilan yang pernah dialami pasien.
  - `Glucose`: Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral.
  - `BloodPressure`: Tekanan darah diastolik (mm Hg).
  - `SkinThickness`: Ketebalan lipatan kulit trisep (mm).
  - `Insulin`: Kadar insulin serum 2 jam (mu U/ml).
  - `BMI`: Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)^2).
  - `DiabetesPedigreeFunction`: Fungsi silsilah diabetes, yang menunjukkan riwayat diabetes dalam keluarga.
  - `Age`: Usia pasien (tahun).
  - `Outcome`: Variabel kelas (0 = tidak diabetes, 1 = diabetes). Ini adalah variabel target yang akan diprediksi.

**Exploratory Data Analysis (EDA):**
Berdasarkan eksplorasi pada notebook, beberapa tahapan penting dalam memahami data telah dilakukan:

  * **Inspeksi Data Awal:** Memuat data ke dalam Pandas DataFrame dan menggunakan `df.head()`, `df.info()`, dan `df.describe()` untuk mendapatkan gambaran umum tentang struktur, tipe data, dan statistik deskriptif. Ditemukan bahwa beberapa kolom numerik seperti `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` mengandung nilai '0' yang tidak masuk akal secara medis, mengindikasikan *missing values*.
  * **Analisis Distribusi Variabel Target:** Memeriksa nilai unik dan hitungan pada kolom `Outcome` menunjukkan bahwa ada ketidakseimbangan kelas, dengan kelas '0' (tidak diabetes) memiliki 500 sampel dan kelas '1' (diabetes) memiliki 268 sampel. Ini akan memerlukan penanganan khusus selama persiapan data untuk mencegah bias model.
  * **Visualisasi Distribusi Fitur:** Histogram dan box plot digunakan untuk memvisualisasikan distribusi masing-masing fitur. Visualisasi ini mengkonfirmasi keberadaan nilai '0' yang tidak valid dan mengidentifikasi potensi *outliers* pada beberapa fitur, seperti `Insulin` dan `Pregnancies`, yang dapat mempengaruhi kinerja model.
  * **Analisis Korelasi:** Heatmap korelasi dibuat untuk melihat hubungan antara semua fitur, termasuk variabel target. Terlihat bahwa fitur `Glucose` dan `BMI` memiliki korelasi positif yang relatif lebih kuat dengan variabel `Outcome`, menunjukkan bahwa kedua fitur ini mungkin merupakan prediktor penting.

**Exploratory Data Analysis (EDA):**
Berdasarkan eksplorasi pada *notebook*, beberapa tahapan penting dalam memahami data telah dilakukan:

### 1.2 Exploratory Data Analysis - Deskripsi Variabel

Pada tahap ini, `df.info()` digunakan untuk melihat dan memahami tipe data, distribusi, dan statistik deskriptif dataset.
```python
# Menampilkan info dataset
df.info()
```
🔍 **Insight Deskripsi Variabel**

Dataset ini terdiri dari **768 sampel (baris) dan 9 fitur (kolom)**. Berikut adalah deskripsi untuk setiap kolom:

* 🤰 **Pregnancies**: Jumlah kehamilan yang pernah dialami oleh pasien. Fitur numerik dengan tipe data integer.
* 🍬 **Glucose**: Konsentrasi glukosa plasma dalam darah setelah 2 jam dalam tes toleransi glukosa oral. Fitur numerik dengan tipe data integer.
* 💓 **BloodPressure**: Tekanan darah diastolik (mm Hg). Fitur numerik dengan tipe data integer.
* 📏 **SkinThickness**: Ketebalan lipatan kulit trisep (mm). Fitur numerik dengan tipe data integer.
* 💉 **Insulin**: Kadar insulin serum setelah 2 jam dalam tes toleransi glukosa oral (mu U/ml). Fitur numerik dengan tipe data integer.
* ⚖️ **BMI**: Indeks Massa Tubuh (Body Mass Index) dihitung dengan berat dalam kg / (tinggi dalam meter)^2. Fitur numerik dengan tipe data float.
* 🧬 **DiabetesPedigreeFunction**: Fungsi riwayat keluarga diabetes. Fitur numerik dengan tipe data float.
* 🎂 **Age**: Usia pasien (tahun). Fitur numerik dengan tipe data integer.
* ✅ **Outcome**: Variabel target yang menunjukkan apakah pasien didiagnosis diabetes (1) atau tidak (0). Fitur kategorikal (biner) dengan tipe data integer.

### 1.3 Exploratory Data Analysis - Univariate Analysis

Pada tahap ini, *Univariate Analysis* dilakukan untuk memahami karakteristik setiap fitur secara individual, dengan menghitung Statistik Deskriptif dan menggunakan visualisasi Histogram.

#### 1.3.1 Statistik Deskriptif

Ringkasan statistik deskriptif menggunakan `.describe()`.
```python
# Menampilkan statistik deskriptif
df.describe()
```
🔍 **Insight Statistik Deskriptif**

* 🤰 **Pregnancies**: Rata-rata **3.85**, min **0**, max **17**. Nilai **0** mungkin tidak valid.
* 🍬 **Glucose**: Rata-rata **120.89**, min **0**, max **199**. Nilai **0** kemungkinan tidak valid.
* 💓 **BloodPressure**: Rata-rata **69.11**, min **0**, max **122**. Nilai **0** kemungkinan tidak valid.
* 📏 **SkinThickness**: Rata-rata **20.54**, min **0**, max **99**. Nilai **0** kemungkinan tidak valid.
* 💉 **Insulin**: Rata-rata **79.80**, min **0**, max **846**. Rentang nilai lebar, kemungkinan *outlier*, dan banyak nilai **0**.
* ⚖️ **BMI**: Rata-rata **31.99**, min **0**, max **67.1**. Nilai **0** kemungkinan tidak valid.
* 🧬 **DiabetesPedigreeFunction**: Rata-rata **0.47**, min **0.08**, max **2.42**.
* 🎂 **Age**: Rata-rata **33.24 tahun**, termuda **21 tahun**, tertua **81 tahun**.
* ✅ **Outcome**: Sekitar **34.9% pasien** (268 dari 768) didiagnosis diabetes (**Outcome = 1**), menunjukkan **ketidakseimbangan kelas**.

📌 **Kesimpulan Statistik Deskriptif:**
Nilai **0** yang tidak realistis pada `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` perlu ditangani. Ketidakseimbangan kelas juga perlu diperhatikan.

#### 1.3.2 Histogram

Histogram digunakan untuk memvisualisasikan distribusi data.
```python
# Histogram Univariate Analysis

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
axes = axes.flatten()

for i, col in enumerate(df.drop('Outcome', axis=1).columns):
    sns.histplot(df[col], ax=axes[i], kde=True, color=sns.color_palette()[i])
    axes[i].set_title(f'Distribution of {col}')

for i in range(len(df.drop('Outcome', axis=1).columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
```
[Gambar Visualisasi Histogram Univariate Analysis]

🔍 **Insight Histogram**

* 🤰 **Pregnancies, SkinThickness, Insulin, DiabetesPedigreeFunction, Age**: Distribusi menceng ke kanan.
* 🍬 **Glucose, BloodPressure, BMI**: Distribusi mendekati normal, tetapi terdapat nilai **0** yang tidak realistis.

📌 **Kesimpulan Histogram:**
Beberapa fitur memiliki distribusi yang menceng, dan nilai **0** yang tidak valid pada `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` perlu ditangani.

```python
# Visualisasi data untuk label outcome
outcome_counts = df['Outcome'].value_counts()
labels = ['No Diabetes', 'Diabetes']
sizes = outcome_counts.values
colors = sns.color_palette('pastel')

# Membuat pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, p * sum(sizes) / 100), startangle=90,
        textprops={'fontsize': 12})
plt.title('Outcome Distribution', fontsize=14)
plt.show()
```
[Gambar Visualisasi Pie Chart Outcome Distribution]

### 1.4 Exploratory Data Analysis - Multivariate Analysis

*Multivariate Analysis* dilakukan untuk memahami hubungan antar fitur menggunakan *Pair Plot* dan *Correlation Matrix*.

#### 1.4.1 Pair Plot

*Pair plot* menampilkan *scatter plot* dan histogram untuk setiap pasangan fitur.
```python
# Pairplot
sns.pairplot(df, hue='Outcome', diag_kind='kde')
plt.show()
```
[Gambar Visualisasi Pair Plot]

🔍 **Insight Pair Plot**

* 🍬✅ **Hubungan Glucose, Age, dan BMI dengan Outcome**: Pasien dengan nilai yang lebih tinggi pada fitur-fitur ini cenderung didiagnosis diabetes.
* 💉🍬 **Hubungan Insulin dan Glucose**: Terdapat korelasi positif, namun tidak terlalu kuat.
* 🤰🎂 **Hubungan Pregnancies dan Age**: Terdapat korelasi positif.
* 🔗 **Korelasi antar Fitur Lain**: Tampak lemah atau tidak signifikan.

📌 **Kesimpulan Pair Plot:**
Fitur **Glucose**, **Age**, dan **BMI** memiliki hubungan yang cukup kuat dengan **Outcome** dan berpotensi menjadi prediktor yang baik.

#### 1.4.2 Correlation Matrix

*Correlation matrix* menunjukkan korelasi antar fitur.
```python
# Correlation Matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Diabetes Dataset Features')
plt.show()
```
[Gambar Visualisasi Correlation Matrix]

🔍 **Insight Correlation Matrix**

* 🔗 **Korelasi terhadap Variabel Target (Outcome)**:
    * 🍬 **Glucose (0.47)** memiliki korelasi tertinggi.
    * ⚖️ **BMI (0.29)**, 🎂 **Age (0.24)**, dan 🤰 **Pregnancies (0.22)** juga berkorelasi positif.
    * 💉 **Insulin (0.13)**, 📏 **SkinThickness (0.07)**, dan 💓 **BloodPressure (0.07)** memiliki korelasi sangat rendah.
* 🔗 **Korelasi Antar Variabel Fitur**:
    * 🤰🎂 **Pregnancies dan Age (0.54)** memiliki korelasi cukup kuat.
    * 💉📏 **Insulin dan SkinThickness (0.44)** serta 📏⚖️ **SkinThickness dan BMI (0.39)** memiliki korelasi sedang.
    * Korelasi antar fitur lainnya tergolong rendah.

📌 **Kesimpulan Correlation Matrix**
Fitur **Glucose**, **BMI**, **Age**, dan **Pregnancies** adalah kandidat utama dalam pemodelan.

### 1.5 Identifikasi Potensi Masalah

Berdasarkan hasil EDA, beberapa potensi masalah teridentifikasi:

- **Nilai 0 yang Tidak Realistis:** Fitur `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` memiliki nilai **0** yang tidak realistis secara medis.
- **Outliers:** Beberapa fitur (`Insulin`, `SkinThickness`, dan `Pregnancies`) memiliki *outliers*.
- **Data Imbalance:** Terdapat ketidakseimbangan jumlah data antara kelas 0 (tidak diabetes) dan kelas 1 (diabetes) pada variabel target (`Outcome`).
- **Distribusi Data yang Miring:** Fitur `Pregnancies`, `SkinThickness`, `Insulin`, dan `DiabetesPedigreeFunction` memiliki distribusi yang menceng (skewed).
- **Multikolinearitas:** Terdapat beberapa korelasi antar fitur, seperti antara `Pregnancies` dan `Age`, serta antara `Insulin` dan `SkinThickness`.

Tentu, saya akan menjelaskan bagian "Data Preparation" secara rinci namun singkat, termasuk alasan mengapa setiap tahapan diperlukan, dan memastikan angka-angka sesuai dengan *notebook*.

## **2. Data Preparation**

Pada tahap ini, data dipersiapkan agar siap digunakan untuk pemodelan *machine learning*.

### 2.1 Data Cleaning

#### 2.1.1 Pengecekan dan Penanganan Duplikat

**Proses:** Memeriksa dan menghapus data duplikat dalam DataFrame menggunakan `.duplicated()`.
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
**Alasan:** Data duplikat dapat menyebabkan model menjadi bias dan *overfitting*. Penghapusan duplikat memastikan setiap observasi unik dan tidak ada pengaruh ganda dari data yang sama. Dalam proyek ini, tidak ditemukan data duplikat.

#### 2.1.2 Pengecekan dan Penanganan Missing Values

**Proses:** Memeriksa jumlah *missing values* (NaN) di setiap kolom DataFrame menggunakan `.isnull().sum()`.
```python
# Periksa missing values
df.isnull().sum()
```
**Alasan:** *Missing values* dapat menyebabkan error pada proses pemodelan atau menghasilkan model yang kurang akurat. Pengecekan ini memastikan tidak ada nilai NaN eksplisit yang perlu ditangani. Pada proyek ini, tidak ditemukan nilai NaN.

#### 2.1.3 Pengecekan dan Penanganan Zero Values (Nilai 0 tidak realistis)

**Proses:** Memeriksa jumlah *zero values* pada fitur-fitur tertentu (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`). Kemudian, nilai 0 ini diganti dengan nilai rata-rata (*mean*) dari masing-masing fitur yang dikelompokkan berdasarkan `Outcome`. Data yang telah diimputasi disimpan ke `df_imputed`.
```python
# Periksa zero values pada fitur-fitur tertentu
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feature in zero_features:
    num_zeros = (df[feature] == 0).sum()
    print(f"Jumlah nilai 0 tidak realistis pada {feature}: {num_zeros}")

# Ganti nilai 0 dengan mean berdasarkan Outcome
df['Glucose'] = df['Glucose'].where((df['Glucose'] > 0)).fillna(df.groupby('Outcome')["Glucose"].transform("mean"))
df['BMI'] = df['BMI'].where((df['BMI'] > 0)).fillna(df.groupby('Outcome')["BMI"].transform("mean"))
df['BloodPressure'] = df['BloodPressure'].where((df['BloodPressure'] > 0)).fillna(df.groupby('Outcome')["BloodPressure"].transform("mean"))
df['Insulin'] = df['Insulin'].where((df['Insulin'] > 0)).fillna(df.groupby('Outcome')["Insulin"].transform("mean"))
df['SkinThickness'] = df['SkinThickness'].where((df['SkinThickness'] > 0)).fillna(df.groupby('Outcome')["SkinThickness"].transform("mean"))

# Simpan ke df_imputed
df_imputed = df.copy()

# Periksa ulang zero values setelah imputasi
for feature in zero_features:
    num_zeros = (df_imputed[feature] == 0).sum()
    print(f"Jumlah nilai 0 tidak realistis pada {feature}: {num_zeros}")
```
**Alasan:** Beberapa fitur memiliki nilai **0** yang tidak realistis secara medis (misalnya, tekanan darah 0 atau BMI 0). Nilai-nilai ini dianggap sebagai *missing values* dan perlu ditangani karena dapat mengganggu perhitungan statistik dan kinerja model. Penggantian dengan *mean* per grup `Outcome` dipilih untuk mempertahankan distribusi data sebaik mungkin dan memanfaatkan informasi dari variabel target. Setelah proses ini, jumlah nilai 0 yang tidak realistis pada `Glucose` adalah 0, `BloodPressure` adalah 0, `SkinThickness` adalah 0, `Insulin` adalah 0, dan `BMI` adalah 0.

#### 2.1.4 Penanganan Outliers dan Distribusi Data yang Miring

**Proses:**
1.  **Identifikasi Outlier:** Menggunakan visualisasi Boxplot untuk mendeteksi keberadaan *outlier* pada setiap fitur numerik.
    ```python
    # Periksa distribusi variabel menggunakan Boxplot
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
    axes = axes.flatten()

    for i, col in enumerate(df_imputed.drop('Outcome', axis=1).columns):
        sns.boxplot(df_imputed[col], ax=axes[i], color=sns.color_palette()[i])
        axes[i].set_title(f'Distribution of {col}')

    for i in range(len(df_imputed.drop('Outcome', axis=1).columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    ```
    [Gambar Visualisasi Boxplot setelah Imputasi]

2.  **Capping (Winsorizing):** Menggunakan fungsi `outlier_thresholds` (berdasarkan metode IQR) untuk menghitung batas bawah dan batas atas. Kemudian, nilai-nilai *outlier* yang berada di luar batas tersebut diganti dengan nilai ambang batas yang sesuai menggunakan fungsi `replace_with_thresholds`. Proses ini diterapkan pada semua kolom numerik. Data disimpan di `df_capped`.
    ```python
    # Fungsi untuk menghitung batas bawah dan atas dengan IQR
    def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        iqr = quartile3 - quartile1
        low_limit = quartile1 - 1.5 * iqr
        up_limit = quartile3 + 1.5 * iqr
        return low_limit, up_limit

    # Fungsi untuk memeriksa apakah kolom memiliki outlier
    def check_outlier(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        return dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None)

    # Fungsi untuk mengganti nilai outlier dan mencetak jumlah yang diganti
    def replace_with_thresholds(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        before = dataframe[col_name].copy()

        # Ganti nilai outlier
        dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
        dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit

        # Hitung berapa yang diganti
        after = dataframe[col_name]
        n_capped = sum(before != after)

        if n_capped > 0:
            print(f"Kolom '{col_name}': {n_capped} nilai dicapping ke batas bawah/atas.")
        else:
            print(f"Kolom '{col_name}': tidak ada nilai yang perlu dicapping.")

    # Salin df_imputed
    df_capped = df_imputed.copy()

    # Kolom numerik
    num_cols = df_capped.select_dtypes(include=['int64', 'float64']).columns

    # Proses capping
    for col in num_cols:
        if check_outlier(df_capped, col):
            replace_with_thresholds(df_capped, col)
        else:
            print(f"Kolom '{col}': tidak ditemukan outlier.")
    ```
    Output yang ditampilkan setelah capping adalah:
    * Kolom 'Pregnancies': 13 nilai dicapping ke batas bawah/atas.
    * Kolom 'Glucose': tidak ditemukan outlier.
    * Kolom 'BloodPressure': tidak ditemukan outlier.
    * Kolom 'SkinThickness': tidak ditemukan outlier.
    * Kolom 'Insulin': 14 nilai dicapping ke batas bawah/atas.
    * Kolom 'BMI': 8 nilai dicapping ke batas bawah/atas.
    * Kolom 'DiabetesPedigreeFunction': 29 nilai dicapping ke batas bawah/atas.
    * Kolom 'Age': 20 nilai dicapping ke batas bawah/atas.
    * Kolom 'Outcome': tidak ditemukan outlier.

3.  **Pemeriksaan Ulang:** Visualisasi boxplot dan histogram diperiksa kembali untuk memastikan *outlier* telah ditangani dan distribusi data terlihat lebih baik. Statistik deskriptif juga ditampilkan kembali.
    ```python
    # Periksa ulang distribusi variabel menggunakan Boxplot
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
    axes = axes.flatten()

    for i, col in enumerate(df_capped.drop('Outcome', axis=1).columns):
        sns.boxplot(df_capped[col], ax=axes[i], color=sns.color_palette()[i])
        axes[i].set_title(f'Distribution of {col}')

    for i in range(len(df_capped.drop('Outcome', axis=1).columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    # Cek ulang distribusi variabel menggunakan Histogram
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
    axes = axes.flatten()

    for i, col in enumerate(df_capped.drop('Outcome', axis=1).columns):
        sns.histplot(df_capped[col], ax=axes[i], kde=True, color=sns.color_palette()[i])
        axes[i].set_title(f'Distribution of {col}')

    for i in range(len(df_capped.drop('Outcome', axis=1).columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    # Cek Ulang Statistik Deskriptif
    df_capped.describe()
    ```
    [Gambar Visualisasi Boxplot setelah Capping]
    [Gambar Visualisasi Histogram setelah Capping]
    [Tabel Statistik Deskriptif setelah Capping]

**Alasan:** *Outliers* dapat memengaruhi kinerja model secara signifikan karena model cenderung terlalu sensitif terhadap nilai ekstrem. Distribusi data yang miring dapat melanggar asumsi beberapa algoritma *machine learning*. *Capping* membantu mengurangi dampak *outlier* tanpa menghapus data, sehingga mempertahankan informasi penting.

### 2.2 Feature Engineering

**Proses:** Menciptakan fitur-fitur baru dari fitur-fitur yang sudah ada.
1.  **Glucose Insulin Ratio:** Fitur baru ini dihitung sebagai rasio antara kadar `Glucose` dan `Insulin`.
    ```python
    df_capped['Glucose_Insulin_Ratio'] = df_capped['Glucose'] / df_capped['Insulin']
    ```
2.  **Binning Fitur Numerik:** Fitur `Glucose`, `Insulin`, `BloodPressure`, `BMI`, dan `Age` dikelompokkan ke dalam kategori berdasarkan referensi klinis atau distribusi data.
    * `Glucose_Group`: Kategorinya adalah 'Rendah', 'Normal', 'Pradiabetes', 'Diabetes'.
    * `Insulin_Group`: Kategorinya adalah 'Rendah', 'Normal', 'Tinggi'.
    * `BloodPressure_Group`: Kategorinya adalah 'Rendah', 'Normal', 'Pra-Hipertensi', 'Hipertensi'.
    * `BMI_Group`: Kategorinya adalah 'Kurus', 'Normal', 'Gemuk', 'Obesitas'.
    * `Age_Group`: Kategorinya adalah 'Muda', 'Dewasa', 'Tua'.
    ```python
    # Fungsi binning
    def bin_glucose(glucose):
        if glucose < 70: return 'Rendah'
        elif 70 <= glucose <= 99: return 'Normal'
        elif 100 <= glucose <= 125: return 'Pradiabetes'
        else: return 'Diabetes'
    df_capped['Glucose_Group'] = df_capped['Glucose'].apply(bin_glucose)

    def bin_insulin(insulin):
        if insulin < 2: return 'Rendah'
        elif 2 <= insulin <= 24: return 'Normal'
        else: return 'Tinggi'
    df_capped['Insulin_Group'] = df_capped['Insulin'].apply(bin_insulin)

    def bin_blood_pressure(bp):
        if bp < 60: return 'Rendah'
        elif 60 <= bp <= 79: return 'Normal'
        elif 80 <= bp <= 89: return 'Pra-Hipertensi'
        else: return 'Hipertensi'
    df_capped['BloodPressure_Group'] = df_capped['BloodPressure'].apply(bin_blood_pressure)

    def bin_bmi(bmi):
        if bmi < 18.5: return 'Kurus'
        elif 18.5 <= bmi <= 24.9: return 'Normal'
        elif 25.0 <= bmi <= 29.9: return 'Gemuk'
        else: return 'Obesitas'
    df_capped['BMI_Group'] = df_capped['BMI'].apply(bin_bmi)

    def bin_age(age):
        if age <= 24: return "Muda"
        elif age <= 41: return "Dewasa"
        else: return "Tua"
    df_capped['Age_Group'] = df_capped['Age'].apply(bin_age)

    df_capped.head()
    ```
    [Gambar Tampilan Awal DataFrame setelah Feature Engineering]

**Alasan:** *Feature Engineering* bertujuan untuk memberikan informasi tambahan kepada model yang mungkin tidak terlihat dari fitur asli. `Glucose_Insulin_Ratio` dapat memberikan wawasan tentang sensitivitas insulin. *Binning* dapat membantu mengubah hubungan non-linear menjadi linear dan mengurangi sensitivitas terhadap *outlier*, serta menambahkan konteks klinis.

### 2.3 Data Encoding

**Proses:** Mengubah fitur-fitur kategorikal yang dibuat pada tahap *Feature Engineering* menjadi format numerik menggunakan **One-Hot Encoding**. Kolom-kolom kategorikal (`Glucose_Group`, `Insulin_Group`, `BloodPressure_Group`, `BMI_Group`, `Age_Group`) di-encode menggunakan `OneHotEncoder`. Hasilnya kemudian digabungkan dengan DataFrame asli yang kolom kategorikalnya telah dihapus, menghasilkan `df_encoded`.
```python
# One-Hot Encoding df_capped
categorical_cols = ['Glucose_Group', 'Insulin_Group', 'BloodPressure_Group', 'BMI_Group', 'Age_Group']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(df_capped[categorical_cols])

encoded_data = encoder.transform(df_capped[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df_capped.index)
df_encoded = df_capped.drop(columns=categorical_cols).join(encoded_df)

# Menampilkan 5 data teratas setelah encoding
df_encoded.head()
```
[Gambar Tampilan Awal DataFrame setelah Encoding]

**Alasan:** Sebagian besar algoritma *machine learning* memerlukan input data dalam bentuk numerik. One-Hot Encoding menciptakan kolom biner untuk setiap kategori, memungkinkan model untuk memproses informasi kategorikal tanpa mengasumsikan urutan atau hubungan ordinal.

### 2.4 Data Splitting

**Proses:** Dataset dibagi menjadi data latih (*training set*) dan data uji (*testing set*). `X` adalah fitur (kolom tanpa `Outcome`) dan `y` adalah variabel target (`Outcome`). Data dibagi dengan `test_size=0.2` (20% untuk pengujian) dan `random_state=42`.
```python
# Data Splitting
X = df_encoded.drop('Outcome', axis=1)
y = df_encoded['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menampilkan ukuran masing-masing set
print("Ukuran X_train:", X_train.shape) # Output: Ukuran X_train: (614, 21)
print("Ukuran X_test:", X_test.shape)   # Output: Ukuran X_test: (154, 21)
print("Ukuran y_train:", y_train.shape) # Output: Ukuran y_train: (614,)
print("Ukuran y_test:", y_test.shape)   # Output: Ukuran y_test: (154,)
```
**Alasan:** Membagi data menjadi *training* dan *testing* set sangat penting untuk mengevaluasi kinerja model secara objektif pada data yang belum pernah dilihat. Hal ini mencegah *data leakage* dan memberikan estimasi yang realistis tentang bagaimana model akan bekerja pada data baru.

### 2.5 Normalisasi Data (Data Scaling)

**Proses:** Fitur-fitur numerik (`Pregnancies`, `Glucose`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, `Glucose_Insulin_Ratio`) diskalakan menggunakan `StandardScaler`. `scaler` difit hanya pada data latih (`X_train`) dan kemudian digunakan untuk mentransformasi baik `X_train` maupun `X_test`.
```python
# Inisialisasi StandardScaler
scaler = StandardScaler()

# Tentukan kolom numerik yang akan diskalakan
numerical_cols = ['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_Insulin_Ratio']

# Fit scaler hanya pada data latih
scaler.fit(X_train[numerical_cols])

# Transform data latih dan data uji
X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
```
**Alasan:** Banyak algoritma *machine learning* sensitif terhadap skala fitur. Jika fitur memiliki rentang nilai yang sangat berbeda, fitur dengan rentang besar dapat mendominasi proses pembelajaran. `StandardScaler` menghasilkan data dengan *mean* 0 dan standar deviasi 1, memastikan semua fitur berkontribusi secara proporsional.

### 2.6 Data Balancing (pada data training)

**Proses:** Teknik **SMOTE** (Synthetic Minority Over-sampling Technique) diterapkan pada data latih (`X_train`, `y_train`) untuk menghasilkan sampel sintetis baru bagi kelas minoritas. Hasilnya adalah `X_train_smote` dan `y_train_smote`.
```python
# Inisialisasi objek SMOTE
smote = SMOTE(random_state=42)

# Terapkan SMOTE pada data training
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Menampilkan jumlah data sebelum dan sesudah SMOTE
print("Jumlah data sebelum SMOTE:")
print(y_train.value_counts()) # Output: 0    400, 1    214
print("\nJumlah data sesudah SMOTE:")
print(pd.Series(y_train_smote).value_counts()) # Output: 0    400, 1    400
```
**Alasan:** Dataset memiliki ketidakseimbangan kelas pada variabel target (`Outcome`), di mana kelas non-diabetes (0) lebih banyak daripada kelas diabetes (1). Ketidakseimbangan ini dapat menyebabkan model cenderung memprediksi kelas mayoritas dan kurang akurat dalam memprediksi kelas minoritas. SMOTE membantu memperluas ruang keputusan untuk kelas minoritas tanpa hanya menduplikasi sampel yang sudah ada, sehingga meningkatkan kinerja model dalam mengidentifikasi kelas minoritas. Ini menunjukkan bahwa SMOTE berhasil menyeimbangkan jumlah sampel untuk kedua kelas di data pelatihan.

## **3. Modeling**

Pada tahap ini, model *machine learning* dibangun dan dilatih untuk memprediksi penyakit diabetes berdasarkan data yang telah diproses.

### 3.1 Model Selection

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

### 3.2 Model Training

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
* Accuracy for Decision Tree: **0.8636**
* Accuracy for Random Forest: **0.8766**
* Accuracy for Gradient Boosting: **0.8896**
* Accuracy for Extra Trees: **0.8701**
* Accuracy for AdaBoost: **0.8831**
* Accuracy for LightGBM: **0.8636**
* Accuracy for XGBoost: **0.8571**

### 3.3 Hyperparameter Tuning

*Hyperparameter tuning* dilakukan menggunakan **Grid Search with Cross-Validation (cv=5)** untuk menemukan kombinasi *hyperparameter* terbaik bagi setiap model.

**Parameter Grid** yang digunakan untuk setiap model adalah sebagai berikut:

```python
rf_params = {
    "max_depth": [10, 15, None],
    "max_features": ["sqrt", "log2", 5],
    "min_samples_split": [5, 10],
    "n_estimators": [100, 200]
}

decision_tree_params = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8]
}

gradient_boosting_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

extra_trees_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

adaboost_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

lightgbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [300, 500],
    "colsample_bytree": [0.7, 1],
    "num_leaves": [15, 31],
    "max_depth": [-1, 5, 10],
    "subsample": [0.8, 1.0]
}

xgboost_params = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [5, 8],
    "n_estimators": [100, 200],
    "colsample_bytree": [0.5, 1],
    "subsample": [0.8, 1.0]
}

param_grids = {
    'Decision Tree': decision_tree_params,
    'Random Forest': rf_params,
    'Gradient Boosting': gradient_boosting_params,
    'Extra Trees': extra_trees_params,
    'AdaBoost': adaboost_params,
    'LightGBM': lightgbm_params,
    'XGBoost': xgboost_params
}
```

**Tuning Model:** `GridSearchCV` diinisialisasi dan dilatih pada data latih yang diseimbangkan (`X_train_smote`, `y_train_smote`). Model terbaik dan akurasi pada data uji dicatat.

```python
# Tuning Model
best_models = {}
tuning_results = []

for name, model in models.items():
    print(f"🔧 Tuning {name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_smote, y_train_smote)

    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    # Prediksi pada data test
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # Simpan hasil tuning
    tuning_results.append({
        'Model': name,
        'Best Parameters': grid_search.best_params_,
        'Best CV Score': round(grid_search.best_score_, 4),
        'Test Accuracy': round(test_acc, 4)
    })

    print(f"✅ {name} tuned.")
    print(f"   Best Params: {grid_search.best_params_}")
    print(f"   CV Score: {grid_search.best_score_:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}\n")

# Tampilkan hasil tuning dalam bentuk tabel
tuning_df = pd.DataFrame(tuning_results)
tuning_df = tuning_df.set_index('Model')
print("\n📋 Tuning Summary:")
display(tuning_df)
```
[Tabel Tuning Summary]

Hasil ringkasan *tuning* menunjukkan performa model setelah optimalisasi *hyperparameter*:

| Model             | Best Parameters                                                                                                                    | Best CV Score | Test Accuracy |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------|---------------|---------------|
| Decision Tree     | {'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 2}                                                                   | 0.8854        | 0.8701        |
| Random Forest     | {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 100}                                              | 0.9116        | 0.8636        |
| Gradient Boosting | {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}                                                        | 0.9290        | 0.8701        |
| Extra Trees       | {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 300}                                                                   | 0.9016        | 0.8571        |
| AdaBoost          | {'learning_rate': 1.0, 'n_estimators': 200}                                                                                        | 0.9028        | **0.9026** |
| LightGBM          | {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 500, 'num_leaves': 31, 'subsample': 0.8}              | 0.9302        | 0.8636        |
| XGBoost           | {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}                                | 0.9315        | 0.8571        |


### 3.4 Memilih Best Model untuk Dievaluasi

Model terbaik dipilih berdasarkan `Test Accuracy`.
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
[Tabel Best Model based on Test Accuracy]

Dari hasil *tuning summary*, **AdaBoost** memiliki `Test Accuracy` tertinggi sebesar **0.9026**, menjadikannya model terbaik yang terpilih untuk evaluasi lebih lanjut.

**Alasan Memilih Model Terbaik:**
Model **AdaBoost Classifier** dipilih sebagai model terbaik karena menunjukkan `Test Accuracy` tertinggi yaitu **0.9026** setelah proses *hyperparameter tuning*. Ini menunjukkan bahwa AdaBoost, dengan kemampuannya berfokus pada sampel yang sulit diklasifikasikan, efektif dalam menggeneralisasi pola data diabetes yang telah diproses dan diseimbangkan.

## **4. Model Evaluation**

Setelah model terbaik dipilih, tahap selanjutnya adalah mengevaluasi kinerjanya secara mendalam pada data uji yang belum pernah dilihat. Evaluasi ini menggunakan metrik yang relevan untuk masalah klasifikasi biner, khususnya dengan mempertimbangkan ketidakseimbangan kelas.

### 4.1 Membuat Prediksi dengan Model Terbaik

Model terbaik yang terpilih adalah **AdaBoost Classifier**. Prediksi dilakukan pada data uji (`X_test`) menggunakan model ini. Probabilitas kelas positif (`y_pred_proba`) juga diperoleh untuk perhitungan ROC AUC.

```python
# Melakukan prediksi pada data uji
y_pred = final_model.predict(X_test)

# Untuk ROC AUC, butuh probabilitas kelas positif
if hasattr(final_model, "predict_proba"):
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
else:
    # Fallback jika predict_proba tidak tersedia
    y_pred_proba = y_pred
    print(f"Peringatan: {best_model_name}.predict_proba() tidak tersedia. ROC AUC mungkin tidak akurat.")

print(f"Prediksi dengan {best_model_name} telah dibuat.")
```

### 4.2 Akurasi dan Confusion Matrix untuk final_model (AdaBoost)

#### Penjelasan Metrik:

* **Akurasi (Accuracy)**:
    * **Formula:** $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
    * **Cara Kerja:** Akurasi mengukur proporsi total prediksi yang benar (baik *True Positive* maupun *True Negative*) dari seluruh jumlah sampel. Ini adalah metrik yang paling umum, tetapi bisa menyesatkan pada dataset dengan ketidakseimbangan kelas, karena model mungkin hanya pandai memprediksi kelas mayoritas.
* **Confusion Matrix**:
    * **Cara Kerja:** *Confusion Matrix* adalah tabel yang menggambarkan kinerja model klasifikasi pada sekumpulan data uji yang hasilnya diketahui. Ini memungkinkan visualisasi kinerja algoritma.
        * **True Positive (TP)**: Jumlah prediksi yang benar bahwa sampel adalah positif (pasien diabetes diprediksi diabetes).
        * **True Negative (TN)**: Jumlah prediksi yang benar bahwa sampel adalah negatif (pasien tidak diabetes diprediksi tidak diabetes).
        * **False Positive (FP)**: Jumlah prediksi yang salah bahwa sampel adalah positif (pasien tidak diabetes diprediksi diabetes - Tipe I Error).
        * **False Negative (FN)**: Jumlah prediksi yang salah bahwa sampel adalah negatif (pasien diabetes diprediksi tidak diabetes - Tipe II Error).

#### Hasil Proyek Berdasarkan Metrik:

**Akurasi:**
Akurasi model AdaBoost pada data uji adalah **0.9026**.
```python
# Menghitung Akurasi (sebagai konfirmasi)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi {best_model_name} pada Data Uji: {accuracy:.4f}")
print("-" * 70)
```
Output: `🎯 Akurasi AdaBoost pada Data Uji: 0.9026`

**Confusion Matrix:**
*Confusion Matrix* untuk model AdaBoost adalah sebagai berikut:
```
[[92  7]
 [ 8 47]]
```
```python
# Menghitung dan Menampilkan Confusion Matrix
print(f"📊 Confusion Matrix untuk {best_model_name}:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("-" * 70)

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Tidak Diabetes', 'Diabetes'],
            yticklabels=['Tidak Diabetes', 'Diabetes'])
plt.xlabel('Prediksi Label')
plt.ylabel('Label Sebenarnya')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.show()
```
[Gambar Visualisasi Confusion Matrix - AdaBoost]

Dari *Confusion Matrix*, kita bisa menguraikan:
* **True Negative (TN)**: **92** (Pasien tidak diabetes diprediksi tidak diabetes)
* **False Positive (FP)**: **7** (Pasien tidak diabetes diprediksi diabetes)
* **False Negative (FN)**: **8** (Pasien diabetes diprediksi tidak diabetes)
* **True Positive (TP)**: **47** (Pasien diabetes diprediksi diabetes)

### 4.3 Laporan Klasifikasi (Classification Report) untuk final_model (AdaBoost)

#### Penjelasan Metrik:

* **Presisi (Precision) - untuk setiap kelas**:
    * **Formula:** $Precision = \frac{TP}{TP + FP}$
    * **Cara Kerja:** Presisi mengukur seberapa banyak prediksi positif yang benar dari total prediksi positif yang dibuat oleh model. Ini penting ketika biaya *False Positive* tinggi (misalnya, diagnosis diabetes yang salah pada orang sehat).
* **Recall (Sensitivitas/True Positive Rate) - untuk setiap kelas**:
    * **Formula:** $Recall = \frac{TP}{TP + FN}$
    * **Cara Kerja:** *Recall* mengukur seberapa banyak sampel positif sebenarnya yang berhasil diidentifikasi oleh model. Ini penting ketika biaya *False Negative* tinggi (misalnya, gagal mendiagnosis diabetes pada pasien yang sebenarnya menderita).
* **F1-Score - untuk setiap kelas**:
    * **Formula:** $F1-Score = 2 * \frac{Precision * Recall}{Precision + Recall}$
    * **Cara Kerja:** F1-Score adalah rata-rata harmonik dari *Precision* dan *Recall*. Ini adalah metrik yang baik untuk digunakan pada dataset dengan ketidakseimbangan kelas karena mempertimbangkan baik *False Positives* maupun *False Negatives*.
* **Support**: Jumlah sampel sebenarnya untuk setiap kelas di data uji.

#### Hasil Proyek Berdasarkan Metrik:

**Laporan Klasifikasi:**
```python
# Menampilkan Laporan Klasifikasi
print(f"📄 Laporan Klasifikasi untuk {best_model_name}:")
report = classification_report(y_test, y_pred, target_names=['Tidak Diabetes (0)', 'Diabetes (1)'])
print(report)
```
Output:
```
📄 Laporan Klasifikasi untuk AdaBoost:
                    precision    recall  f1-score   support

Tidak Diabetes (0)       0.92      0.93      0.93        99
    Diabetes (1)       0.87      0.85      0.86        55

        accuracy                           0.90       154
       macro avg       0.89      0.89      0.89       154
    weighted avg       0.90      0.90      0.90       154
```
Dari Laporan Klasifikasi:

* **Kelas 'Tidak Diabetes (0)'**:
    * Precision: **0.92**
    * Recall: **0.93**
    * F1-Score: **0.93**
    * Support: **99**
* **Kelas 'Diabetes (1)'**:
    * Precision: **0.87**
    * Recall: **0.85**
    * F1-Score: **0.86**
    * Support: **55**
* **Accuracy**: **0.90**
* **Macro Avg F1-Score**: **0.89**
* **Weighted Avg F1-Score**: **0.90**

Untuk konteks masalah prediksi diabetes, *Recall* untuk kelas 'Diabetes (1)' sangat penting. Nilai *Recall* **0.85** berarti model berhasil mengidentifikasi 85% dari semua pasien yang sebenarnya menderita diabetes, yang merupakan hasil yang baik untuk mengurangi *False Negative* (pasien diabetes yang tidak terdeteksi).

### 4.4 ROC AUC Score dan Kurva ROC untuk final_model (AdaBoost)

#### Penjelasan Metrik:

* **ROC AUC Score (Receiver Operating Characteristic - Area Under the Curve)**:
    * **Formula:** Area di bawah kurva ROC.
    * **Cara Kerja:** Kurva ROC memplot *True Positive Rate* (TPR atau *Recall*) terhadap *False Positive Rate* (FPR atau 1 - *Specificity*) pada berbagai ambang batas klasifikasi.
        * $TPR = \frac{TP}{TP + FN}$
        * $FPR = \frac{FP}{FP + TN}$
    * AUC mengukur kemampuan model untuk membedakan antara kelas positif dan negatif. Nilai AUC berkisar dari 0 hingga 1. Semakin tinggi nilai AUC, semakin baik model dalam membedakan kelas. AUC **0.5** menunjukkan kinerja acak, sementara **1.0** menunjukkan pembeda yang sempurna.

#### Hasil Proyek Berdasarkan Metrik:

**ROC AUC Score:**
ROC AUC Score untuk model AdaBoost adalah **0.9419**.
```python
# Menghitung dan Menampilkan ROC AUC Score
if hasattr(final_model, "predict_proba"):
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"📈 ROC AUC Score untuk {best_model_name}: {roc_auc:.4f}")
    print("-" * 70)

    # Visualisasi Kurva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{best_model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--') # Garis referensi (random guess)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity/Recall)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {best_model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
else:
    print(f"Kurva ROC tidak dapat ditampilkan untuk {best_model_name} karena predict_proba tidak tersedia.")
```
[Gambar Visualisasi Receiver Operating Characteristic (ROC) Curve - AdaBoost]

ROC AUC Score sebesar **0.9419** menunjukkan bahwa model AdaBoost memiliki kemampuan diskriminasi yang sangat baik dalam membedakan antara pasien yang menderita diabetes dan tidak. Nilai ini mendekati 1, mengindikasikan bahwa model memiliki probabilitas tinggi untuk memberikan peringkat yang lebih tinggi pada sampel positif (diabetes) dibandingkan sampel negatif (tidak diabetes) secara acak. Kurva ROC juga divisualisasikan, menunjukkan seberapa baik model menyeimbangkan *True Positive Rate* dan *False Positive Rate* di berbagai ambang batas.

**Referensi:**

[1] World Health Organization. (2021). *Diabetes*. Retrieved from [https://www.who.int/news-room/fact-sheets/detail/diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)/n
[2] American Diabetes Association. (2020). *Statistics About Diabetes*. Retrieved from [https://diabetes.org/about-diabetes/statistics/about-diabetes](https://diabetes.org/about-diabetes/statistics/about-diabetes)
[3] Reza, M. S., Amin, R., Yasmin, R., Kulsum, W., & Ruhi, S. (2024). Improving diabetes disease patients classification using stacking ensemble method with PIMA and local healthcare data. *Heliyon*, *10*(7), e24536. [https://doi.org/10.1016/j.heliyon.2024.e24536](https://www.google.com/search?q=https://doi.org/10.1016/j.heliyon.2024.e24536)
[4] International Diabetes Federation. (2021). *IDF Diabetes Atlas, 10th Edition*. Brussels, Belgium: International Diabetes Federation. [https://diabetesatlas.org/](https://diabetesatlas.org/)
[5] Sarwar, T., Iqbal, A., & Bashir, T. (2020). Machine learning approach for diabetes prediction using Pima Indian diabetes dataset. *International Journal of Advanced Computer Science and Applications, 11*(11). [https://doi.org/10.14569/IJACSA.2020.0111162](https://www.google.com/search?q=https://doi.org/10.14569/IJACSA.2020.0111162)
[6] Ahmed, A., Khan, J., Arsalan, M., Ahmed, K., Shahat, A. A., Alhalmi, A., & Naaz, S. (2025). Machine Learning Algorithm-Based Prediction of Diabetes Among Female Population Using PIMA Dataset. *Healthcare*, *13*(1), 37. [https://doi.org/10.3390/healthcare13010037](https://www.google.com/search?q=https://doi.org/10.3390/healthcare13010037)
[7] Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann. (Link umum ke buku: [https://www.elsevier.com/books/data-mining-concepts-and-techniques/han/978-0-12-381479-1](https://www.elsevier.com/books/data-mining-concepts-and-techniques/han/978-0-12-381479-1))
