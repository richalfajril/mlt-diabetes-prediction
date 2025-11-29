import streamlit as st
import joblib
import pandas as pd

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Sistem Prediksi Diabetes",
    page_icon="🏥",
    layout="wide"
)

# ==============================
# LOAD MODEL & SCALER
# ==============================
@st.cache_resource
def load_models():
    try:
        model = joblib.load("final_diabetes_model.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error saat memuat model: {str(e)}")
        st.stop()

# ==============================
# FUNGSI UTAMA
# ==============================
def main():
    model, scaler = load_models()

    st.title("🏥 Sistem Prediksi Diabetes")
    st.write("Sistem ini memprediksi risiko diabetes berdasarkan data kesehatan pasien.")

    # ==============================
    # INFORMASI DASAR
    # ==============================
    st.subheader("📋 Informasi Dasar")
    col_basic1, col_basic2 = st.columns(2)

    with col_basic1:
        age = st.number_input(
            "Usia",
            min_value=0,
            value=0,
        )

        pregnancies = st.number_input(
            "Jumlah Kehamilan",
            min_value=0,
            value=0,
        )

    with col_basic2:
        has_family_diabetes = st.checkbox("Riwayat Diabetes Keluarga")
        diabetes_pedigree = 0.8 if has_family_diabetes else 0.1

    # ==============================
    # PENGUKURAN FISIK
    # ==============================
    st.subheader("📏 Pengukuran Fisik")
    col_physical1, col_physical2 = st.columns(2)

    with col_physical1:
        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=0.0,
        )

        skin_thickness = st.number_input(
            "Ketebalan Kulit (mm)",
            min_value=0,
            value=0,
        )

    with col_physical2:
        blood_pressure = st.number_input(
            "Tekanan Darah (mm Hg)",
            min_value=0,
            value=0,
        )

    # ==============================
    # HASIL LABORATORIUM
    # ==============================
    st.subheader("🔬 Hasil Laboratorium")
    col_lab1, col_lab2 = st.columns(2)

    with col_lab1:
        glucose = st.number_input(
            "Kadar Glukosa (mg/dL)",
            min_value=0,
            value=0,
        )

    with col_lab2:
        insulin = st.number_input(
            "Kadar Insulin (mu U/ml)",
            min_value=0,
            value=0,
        )

    # ==============================
    # VALIDASI INPUT (KECUALI KEHAMILAN)
    # ==============================
    invalid_inputs = {
        "Usia": age,
        "Glukosa": glucose,
        "Insulin": insulin,
        "BMI": bmi,
        "Tekanan Darah": blood_pressure,
        "Ketebalan Kulit": skin_thickness
    }

    if st.button("🔍 Analisis Risiko Diabetes", type="primary"):

        if any(value == 0 for value in invalid_inputs.values()):
            st.error("❌ Semua input WAJIB diisi dan tidak boleh 0 (kecuali Jumlah Kehamilan).")
            st.stop()

        # ==============================
        # DATA INPUT MODEL
        # ==============================
        input_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age
        }

        input_df = pd.DataFrame([input_data])

        # ==============================
        # FITUR TURUNAN
        # ==============================
        input_df["Glucose_Insulin_Ratio"] = input_df["Glucose"] / input_df["Insulin"]

        # Urutan fitur HARUS sama seperti saat training
        # Scaled features (sesuai numerical_cols di training)
        scaled_features = [
            "Pregnancies",
            "Glucose",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Glucose_Insulin_Ratio"
        ]

        # ==============================
        # SCALING - HANYA SCALED FEATURES
        # ==============================
        input_scaled = scaler.transform(input_df[scaled_features])
        input_scaled_df = pd.DataFrame(input_scaled, columns=scaled_features)

        # ==============================
        # GABUNGKAN DENGAN FITUR TIDAK TERSCALE
        # ==============================
        # Urutan akhir harus: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Glucose_Insulin_Ratio
        input_final = pd.DataFrame()
        input_final["Pregnancies"] = input_scaled_df["Pregnancies"]
        input_final["Glucose"] = input_scaled_df["Glucose"]
        input_final["BloodPressure"] = blood_pressure
        input_final["SkinThickness"] = skin_thickness
        input_final["Insulin"] = input_scaled_df["Insulin"]
        input_final["BMI"] = input_scaled_df["BMI"]
        input_final["DiabetesPedigreeFunction"] = input_scaled_df["DiabetesPedigreeFunction"]
        input_final["Age"] = input_scaled_df["Age"]
        input_final["Glucose_Insulin_Ratio"] = input_scaled_df["Glucose_Insulin_Ratio"]

        # ==============================
        # PREDIKSI
        # ==============================
        try:
            prediction = model.predict(input_final)
            prediction_proba = model.predict_proba(input_final)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.stop()

        # ==============================
        # HASIL
        # ==============================
        st.subheader("🔎 Hasil Analisis")

        if prediction[0] == 1:
            st.error("⚠️ Risiko Tinggi Diabetes")
        else:
            st.success("✅ Risiko Rendah Diabetes")

        st.write(f"**Probabilitas terkena diabetes:** {prediction_proba[0][1]:.2%}")
        st.progress(float(prediction_proba[0][1]))

    # ==============================
    # SIDEBAR INFORMASI
    # ==============================
    with st.sidebar:
        st.title("ℹ️ Informasi")
        with st.expander("Tentang Model Ini"):
            st.write(
                """
Model prediksi diabetes ini menggunakan machine learning untuk menilai 
risiko seseorang berdasarkan beberapa indikator kesehatan.

**Data yang digunakan meliputi:**

1. Informasi Dasar  
   - Usia  
   - Jumlah kehamilan  
   - Riwayat diabetes keluarga  

2. Pengukuran Fisik  
   - Indeks Massa Tubuh (BMI)  
   - Tekanan darah  
   - Ketebalan kulit  

3. Hasil Laboratorium  
   - Kadar glukosa  
   - Kadar insulin  

⚠️ **Catatan:**  
Model ini bukan alat diagnosis medis, hanya sebagai alat bantu prediksi risiko.
Selalu konsultasikan hasil dengan tenaga kesehatan profesional.
"""
        )

# ==============================
# EKSEKUSI APLIKASI
# ==============================
if __name__ == "__main__":
    main()
