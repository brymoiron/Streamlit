import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# ---------------------------
# Load trained model + scaler
# ---------------------------
model = tf.keras.models.load_model("irrigation_lstm.h5", compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
scaler = joblib.load("scaler.pkl")

st.title("🌱 Smart Irrigation - Soil Moisture Prediction")
st.markdown("Enter weather data for the **past 7 days** to predict tomorrow's soil moisture.")

# ---------------------------
# Create 7 days of inputs
# ---------------------------
st.subheader("📅 Enter Weather Data for Past 7 Days")

# Option 1: Use a simple form with same values for all 7 days
with st.expander("Quick Test (Same values for all 7 days)"):
    temp = st.number_input("🌡️ Temperature (°C)", value=30.0, key="temp_quick")
    precip = st.number_input("🌧️ Precipitation (mm)", value=0.0, key="precip_quick")
    humidity = st.number_input("💧 Humidity (%)", value=60.0, key="humidity_quick")
    wind = st.number_input("🌬️ Wind Speed (m/s)", value=5.0, key="wind_quick")
    
    if st.button("Predict with Same Values"):
        # Create 7-day sequence with same values
        X = np.array([[temp, precip, humidity, wind]] * 7)  # shape (7, 4)
        X_scaled = scaler.transform(X)  # scale
        X_seq = X_scaled.reshape(1, 7, 4)  # reshape for LSTM: (1, 7, 4)
        
        # Predict
        prediction = model.predict(X_seq, verbose=0)[0][0]
        
        st.subheader(f"📊 Predicted Soil Moisture: **{prediction:.2f}%**")
        
        # Simple rules
        if prediction < 13:
            st.warning("💧 Soil is dry → Irrigation Required")
        elif prediction > 27:
            st.info("🌧️ Soil too wet → Hold Irrigation")
        else:
            st.success("✅ Soil moisture optimal → No irrigation needed")

# Option 2: Enter different values for each day
st.subheader("Or enter different values for each of the 7 days:")

# Create a dataframe for user input
data = []
for day in range(7, 0, -1):
    st.write(f"**Day {8-day} (T-{day} days ago)**")
    cols = st.columns(4)
    with cols[0]:
        temp = st.number_input(f"Temp (°C)", value=30.0, key=f"temp_{day}")
    with cols[1]:
        precip = st.number_input(f"Precip (mm)", value=0.0, key=f"precip_{day}")
    with cols[2]:
        humidity = st.number_input(f"Humidity (%)", value=60.0, key=f"humidity_{day}")
    with cols[3]:
        wind = st.number_input(f"Wind (m/s)", value=5.0, key=f"wind_{day}")
    
    data.append([temp, precip, humidity, wind])

if st.button("Predict with 7-Day Sequence"):
    # Prepare sequence
    X = np.array(data)  # shape (7, 4)
    X_scaled = scaler.transform(X)  # scale
    X_seq = X_scaled.reshape(1, 7, 4)  # reshape for LSTM
    
    # Predict
    prediction = model.predict(X_seq, verbose=0)[0][0]
    
    st.subheader(f"📊 Predicted Soil Moisture: **{prediction:.2f}%**")
    
    # Show input summary
    st.write("**Input Summary:**")
    df_input = pd.DataFrame(data, columns=["Temperature", "Precipitation", "Humidity", "WindSpeed"])
    df_input.index = [f"Day {i+1}" for i in range(7)]
    st.dataframe(df_input)
    
    # Simple rules
    if prediction < 30:
        st.warning("💧 Soil is dry → Irrigation Required")
    elif prediction > 60:
        st.info("🌧️ Soil too wet → Hold Irrigation")
    else:
        st.success("✅ Soil moisture optimal → No irrigation needed")