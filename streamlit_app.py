import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("Liver_Prediction_SVM.pkl")

label_map = {
    0: "Cirrhosis",
    1: "Fibrosis",
    2: "Hepatitis",
    3: "No Disease",
    4: "Suspected Disease"
}

st.set_page_config(page_title="Liver Disease Prediction", layout="centered")
st.title("🧪 Liver Disease Prediction")

st.write("Enter patient lab values to predict liver condition.")

# ---- Inputs (RAW values) ----
age = st.number_input("Age", min_value=1, max_value=100, value=45)
cholinesterase = st.number_input("Cholinesterase", value=7.0)
cholesterol = st.number_input("Cholesterol", value=4.5)
protein = st.number_input("Total Protein", value=65.0)
albumin = st.number_input("Albumin", value=40.0)

alk_phos = st.number_input("Alkaline Phosphatase", value=90.0)
alt = st.number_input("ALT (SGPT)", value=30.0)
ast = st.number_input("AST (SGOT)", value=35.0)
bilirubin = st.number_input("Bilirubin", value=8.0)
creatinine = st.number_input("Creatinine", value=100.0)
ggt = st.number_input("GGT", value=40.0)

# Albumin outlier logic (same rule used in training)
albumin_high_outlier = 1 if albumin > 50 else 0

if st.button("Predict"):
    # ---- Feature engineering (must match training) ----
    features = np.array([
        age,
        cholinesterase,
        cholesterol,
        protein,
        albumin_high_outlier,
        np.log(alk_phos),
        np.log(alt),
        np.log(ast),
        np.log(bilirubin),
        np.log(creatinine),
        np.log(ggt)
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]

    st.subheader("Prediction Result")
    st.success(f"🩺 {label_map[prediction]}")
