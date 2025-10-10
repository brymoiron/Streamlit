import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# ---------------------------
# Load trained model + scaler
# ---------------------------
model = tf.keras.models.load_model("irrigation_lstm.h5", compile=False)
# Recompile with fresh config
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
scaler = joblib.load("scaler.pkl")

st.title("🌱 Smart Irrigation - Soil Moisture Prediction")
st.markdown("Enter today's weather data and get a prediction of soil moisture levels.")

# ---------------------------
# User inputs
# ---------------------------
temp = st.number_input("🌡️ Temperature (°C)", value=30.0)
precip = st.number_input("🌧️ Precipitation (mm)", value=0.0)
humidity = st.number_input("💧 Humidity (%)", value=60.0)
wind = st.number_input("🌬️ Wind Speed (m/s)", value=5.0)

if st.button("Predict Soil Moisture"):
    # ---------------------------
    # Preprocess input
    # ---------------------------
    X = np.array([[temp, precip, humidity, wind]])   # shape (1,4)
    X_scaled = scaler.transform(X)                   # scale with same scaler

    # LSTM expects (samples, time_steps, features)
    # Here we simulate a 7-day sequence by repeating today’s input 7 times
    X_seq = np.repeat(X_scaled, 7, axis=0).reshape(1, 7, 4)

    # ---------------------------
    # Prediction
    # ---------------------------
    prediction = model.predict(X_seq)[0][0]

    st.subheader(f"Predicted Soil Moisture: **{prediction:.2f}**")

    # ---------------------------
    # Simple expert system rules
    # ---------------------------
    if prediction < 30:
        st.warning("💧 Soil is dry → Irrigation Required")
    elif prediction > 60:
        st.info("🌧️ Soil too wet → Hold Irrigation")
    else:
        st.success("✅ Soil moisture optimal → No irrigation needed")
