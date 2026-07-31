import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

model=joblib.load('LogReg_Heart.pkl')
scaler=joblib.load('scaler.pkl')
Columns=joblib.load('Columns.pkl')

st.title("❤️ Heart Disease Risk Predictor")
st.markdown("### 🩺 Enter patient details below to estimate heart disease risk")
st.markdown("---")

st.subheader("👤 Personal Info")
col1,col2=st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)

with col2:
    sex = st.selectbox("Sex", ["M", "F"])

st.subheader("💓 Vitals")
col3,col4=st.columns(2)
with col3:
    cholesterol = st.number_input("🧈 Cholesterol", min_value=0, value=200)

with col4:
    resting_bp = st.number_input("🩸 Resting Blood Pressure", min_value=0, value=120)

col5,col6=st.columns(2)
with col5:
    max_hr = st.number_input("🏃 Max Heart Rate", min_value=60, max_value=220, value=150)

with col6:
    fasting_bs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", [0, 1])

st.subheader("🫀 Cardiac Symptoms")
chest_pain = st.selectbox("💥 Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])

resting_ecg = st.selectbox("📈 Resting ECG", ["Normal", "ST", "LVH"])
exercise_angina = st.selectbox("🏋️ Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.number_input("📉 Oldpeak", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
st_slope = st.selectbox("⛰️ ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")


if st.button("🔍 Predict Risk", use_container_width=True):
    input_dict = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingECG': resting_ecg,
        'ExerciseAngina': exercise_angina,
        'ST_Slope': st_slope,
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=Columns, fill_value=0)
 
    numeric_col = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_encoded[numeric_col] = scaler.transform(input_encoded[numeric_col])

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
 
    st.markdown("## 📋 Result")

    if prediction == 1:
        st.error(f"⚠️🚨 **High risk of heart disease** — probability: **{probability:.1%}** 💔")
        st.progress(probability)
        st.markdown("🏥 *Please consult a cardiologist for further evaluation.*")
    else:
        st.success(f"✅💚 **Low risk of heart disease** — probability: **{probability:.1%}**")
        st.progress(probability)
        st.markdown("🎉 *Keep up the healthy habits!*")
 
    with st.expander("🔬 See input summary"):
        st.dataframe(input_df)
 
st.markdown("---")
st.caption("🤖 Powered by Logistic Regression | ⚠️ For educational purposes only — not a substitute for professional medical advice.")
