import streamlit as st
import numpy as np
import joblib

# Load model
def load_model():
    return joblib.load("Liver_Prediction_SVM.pkl")

model = load_model()
label_map = {
    0: "Cirrhosis",
    1: "Fibrosis",
    2: "Hepatitis",
    3: "No Disease",
    4: "Suspected Disease"
}


st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("## 🧬 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=45)
    cholinesterase = st.number_input("Cholinesterase (kU/L)", value=7.0)
    cholesterol = st.number_input("Cholesterol (mmol/L)", value=4.5)
    protein = st.number_input("Total Protein (g/L)", value=65.0)

with col2:
    albumin = st.number_input("Albumin (g/L)", value=40.0)
    alk_phos = st.number_input("Alkaline Phosphatase (U/L)", value=90.0)
    alt = st.number_input("ALT / SGPT (U/L)", value=30.0)
    ast = st.number_input("AST / SGOT (U/L)", value=35.0)

st.markdown("## 🧪 Additional Lab Values")

col3, col4 = st.columns(2)

with col3:
    bilirubin = st.number_input("Bilirubin (µmol/L)", value=8.0)
    creatinine = st.number_input("Creatinine (µmol/L)", value=100.0)

with col4:
    ggt = st.number_input("GGT (U/L)", value=40.0)

st.markdown("---")

if st.button("🔍 Predict Liver Condition", use_container_width=True):

    albumin_high_outlier = 1 if albumin > 50 else 0

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
    label = label_map[prediction]

    st.markdown("### 🧾 Prediction Result")

    if label == "No Disease":
        st.success("✅ **No Liver Disease Detected**")
    elif label == "Suspected Disease":
        st.warning("⚠️ **Suspected Liver Disease**")
    else:
        st.error(f"🚨 **{label} Detected**")
with st.expander("ℹ️ About this prediction"):
    st.write("""
    - This model is trained on clinical lab data.
    - It predicts **multiple liver disease categories**.
    - Results are **not a medical diagnosis**.
    - Always consult a qualified healthcare professional.
    """)
with st.sidebar:
    st.markdown("## 🧪 Liver AI")
    st.write("Multi-class liver disease prediction")
    st.markdown("---")
    st.write("Built with:")
    st.write("- scikit-learn")
    st.write("- Streamlit Cloud")
