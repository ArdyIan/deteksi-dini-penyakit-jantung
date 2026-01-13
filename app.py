import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# FUNGSI PEMBANTU (SHAP WRAPPER)
def st_shap(plot, height=None):
    """Fungsi untuk menampilkan plot SHAP JavaScript di Streamlit"""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# 1. LOAD MODEL & SCALER
# Pastikan nama file .pkl sesuai 
try:
    model = joblib.load('model_jantung_best.pkl')
    scaler = joblib.load('scaler_jantung.pkl')
except Exception as e:
    st.error(f"Gagal memuat model/scaler. Error: {e}")


# 2. KONFIGURASI HALAMAN
st.set_page_config(page_title="Heart Discovery AI", layout="centered")
st.title("🏥 Sistem Deteksi Risiko Penyakit Jantung")
st.markdown("Aplikasi skrining awal menggunakan **Logistic Regression** dan **SHAP Explainable AI**.")
st.divider()


# 3. FORM INPUT USER
st.subheader("📋 Data Klinis Pasien")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Jenis Kelamin", ["Laki-laki (M)", "Perempuan (F)"])
        cp_type = st.selectbox("Tipe Nyeri Dada", ["ASY: Asymptomatic", "ATA: Atypical Angina", "NAP: Non-Anginal", "TA: Typical Angina"])
        resting_bp = st.number_input("Tekanan Darah (mm Hg)", min_value=0, max_value=250, value=120)
        cholesterol = st.number_input("Kolesterol Serum (mm/dl)", min_value=0, max_value=700, value=200)

    with col2:
        fasting_bs = st.selectbox("Gula Darah Puasa > 120 mg/dl", ["Tidak", "Ya"])
        resting_ecg = st.selectbox("Hasil EKG Istirahat", ["Normal", "ST: Abnormalitas ST-T", "LVH: Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Detak Jantung Maksimal (bpm)", min_value=50, max_value=250, value=150)
        ex_angina = st.selectbox("Nyeri Dada Saat Olahraga", ["Tidak", "Ya"])
        oldpeak = st.number_input("Depresi ST (Oldpeak)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
        st_slope = st.selectbox("Kemiringan Segmen ST (Slope)", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Analisis Risiko Pasien")


# 4. LOGIKA PEMROSESAN & PREDIKSI
if submitted:
    # A. Preprocessing
    sex_encoded = 1 if "Laki-laki" in sex else 0
    fbs_encoded = 1 if fasting_bs == "Ya" else 0
    ex_angina_encoded = 1 if ex_angina == "Ya" else 0
    slope_map = {"Up": 0, "Flat": 1, "Down": 2}
    st_slope_encoded = slope_map[st_slope]
    
    cp_ATA = 1 if "ATA" in cp_type else 0
    cp_NAP = 1 if "NAP" in cp_type else 0
    cp_TA  = 1 if "TA" in cp_type else 0
    
    ecg_ST  = 1 if "ST" in resting_ecg else 0
    ecg_LVH = 1 if "LVH" in resting_ecg else 0

    chol_final = cholesterol if cholesterol != 0 else 237.0
    chol_missing = 1 if cholesterol == 0 else 0

    # B. Susun Data
    feature_names = [
        'age', 'restingbp', 'cholesterol', 'fastingbs', 'maxhr', 'oldpeak', 
        'sex_encoded', 'cp_ATA', 'cp_NAP', 'cp_TA', 
        'exerciseangina_encoded', 'st_slope_encoded', 'ecg_ST', 'ecg_LVH', 
        'Cholesterol_missing'
    ]
    
    data_user = pd.DataFrame([[
        age, resting_bp, chol_final, fbs_encoded, max_hr, oldpeak,
        sex_encoded, cp_ATA, cp_NAP, cp_TA,
        ex_angina_encoded, st_slope_encoded, ecg_ST, ecg_LVH,
        chol_missing
    ]], columns=feature_names)

    # C. Scaling 
    num_cols = ['age', 'restingbp', 'cholesterol', 'maxhr', 'oldpeak']
    data_user[num_cols] = scaler.transform(data_user[num_cols])

    # D. Prediksi 
    prediction = model.predict(data_user)[0]
    probability = model.predict_proba(data_user)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"### HASIL DIAGNOSA: RISIKO TINGGI (POSITIF)")
        st.write(f"Keyakinan Model: **{probability:.2%}**")
    else:
        st.success(f"### HASIL DIAGNOSA: RISIKO RENDAH (NEGATIF)")
        st.write(f"Keyakinan Model: **{(1-probability):.2%}**")


    # 5. VISUALISASI SHAP (WATERFALL PLOT)
    st.subheader("🔍 Interpretasi SHAP (Waterfall Plot)")
    
    try:
        # A. LOAD DATA REFERENSI (SANGAT PENTING)
        # butuh data ini agar SHAP bisa membandingkan input user dengan rata-rata data training
        @st.cache_data
        def get_background_data():
            df_train = pd.read_csv('encoded_data_train.csv')
            # Ambil hanya fitur (tanpa kolom target 'heartdisease')
            X_train = df_train.drop(columns=['heartdisease'])
            # Ambil 50 baris secara acak sebagai representasi latar belakang
            return X_train.sample(50, random_state=42)

        X_reference = get_background_data()

        # B. INISIALISASI EXPLAINER DENGAN REFERENSI
        # Menggunakan KernelExplainer agar bisa menghitung probabilitas
        def model_predict_proba(data):
            return model.predict_proba(data)[:, 1]

        # SHAP membandingkan input user terhadap X_reference
        explainer = shap.KernelExplainer(model_predict_proba, X_reference)
        shap_values = explainer.shap_values(data_user)

        # C. BUAT OBJEK EXPLANATION
     
        explanation = shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value, 
            data=data_user.iloc[0], 
            feature_names=feature_names
        )

        # D. PLOTTING
        fig, ax = plt.subplots(figsize=(10, 6))

        shap.plots.waterfall(explanation, max_display=10, show=False)
        
        plt.title("Analisis Faktor Risiko Pasien (SHAP Waterfall)", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Gagal memuat Waterfall Plot: {e}")
        st.write("Pastikan file 'encoded_data_train.csv' ada di folder yang sama.")