import streamlit as st
import joblib
from extract_features import extract_vector

st.title("🛡️ Malware Family Classifier (Static Analysis)")
uploaded_file = st.file_uploader("Upload a PE file (.exe)", type=["exe"])

if uploaded_file:
    with open("uploaded_sample.exe", "wb") as f:
        f.write(uploaded_file.read())

    st.info("🔍 Extracting features...")
    try:
        features = extract_vector("uploaded_sample.exe")
        model = joblib.load("malware_model.pkl")
        prediction = model.predict(features)[0]
        st.success(f"✅ Malware family: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Error during classification: {e}")
