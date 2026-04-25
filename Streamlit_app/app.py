import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# Load trained model safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "rf_model.pkl")

# Debug: show available files
st.write("📂 Files in current directory:", os.listdir(BASE_DIR))

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file 'rf_model.pkl' not found. Please upload it to the same folder as app.py.")
    st.stop()

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Crop Prediction", layout="centered")

st.title("🌱 Crop Prediction App")
st.write("Enter soil and environmental details to predict the best crop.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("pH value", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Crop"):
    try:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)

        st.success(f"🌾 Recommended Crop: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
