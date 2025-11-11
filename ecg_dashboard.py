import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load ECG CSV
st.title("📊 ECG Analysis Dashboard")

uploaded_file = st.file_uploader("Upload ECG CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Raw Data")
    st.write(df.head())

    # Assume columns: 'time', 'voltage'
    time = df['time']
    voltage = df['voltage']

    # Plot ECG signal
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, voltage, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('ECG Signal')
    st.pyplot(fig)

    # Generate Summary
    mean_v = np.mean(voltage)
    max_v = np.max(voltage)
    min_v = np.min(voltage)
    std_v = np.std(voltage)

    # Detect R-peaks (using simple peak finder)
    peaks, _ = find_peaks(voltage, distance=50, height=np.mean(voltage) + np.std(voltage))
    num_beats = len(peaks)
    duration_sec = time.iloc[-1] - time.iloc[0]
    heart_rate = (num_beats / duration_sec) * 60 if duration_sec > 0 else 0

    st.subheader("📈 ECG Report Summary")
    st.write(f"**Mean Voltage:** {mean_v:.2f} mV")
    st.write(f"**Max Voltage:** {max_v:.2f} mV")
    st.write(f"**Min Voltage:** {min_v:.2f} mV")
    st.write(f"**Standard Deviation:** {std_v:.2f}")
    st.write(f"**Estimated Heart Rate:** {heart_rate:.1f} BPM")

    # Interpretation
    if heart_rate < 60:
        condition = "Bradycardia (Slow Heart Rate)"
    elif 60 <= heart_rate <= 100:
        condition = "Normal Heart Rate"
    else:
        condition = "Tachycardia (Fast Heart Rate)"

    st.success(f"🩺 Interpretation: {condition}")
