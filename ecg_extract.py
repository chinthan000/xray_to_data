import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load ECG image
img = cv2.imread('ecg_sample.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to highlight ECG trace
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Get coordinates of ECG waveform
points = np.column_stack(np.where(binary > 0))
x = points[:, 1]  # horizontal (time)
y = points[:, 0]  # vertical (voltage)

# Sort points by time (x)
sorted_indices = np.argsort(x)
x = x[sorted_indices]
y = y[sorted_indices]

# ---- STEP 1: Convert pixels → time ----
# Estimate scale: 25 mm/s → if 1 mm = ~40 pixels, then 1000 pixels = 1 sec approx
# You might need to tweak this depending on your image size
# ---- STEP 1: Convert pixels → time ----
time_scale = 1 / 94.5   # seconds per pixel (≈0.0106s)
time = x * time_scale

# ---- STEP 2: Convert pixels → voltage ----
voltage_scale = 1 / 37.8   # mV per pixel (≈0.0264mV)
voltage = -y * voltage_scale
# invert to make upward positive

# ---- STEP 3: Remove duplicates (same time) ----
unique_times, unique_indices = np.unique(time, return_index=True)
time = time[unique_indices]
voltage = voltage[unique_indices]

# ---- STEP 4: Plot and save ----
plt.plot(time, voltage)
plt.title("Extracted ECG Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.show()

# Save to CSV
data = pd.DataFrame({'time': time, 'voltage': voltage})
data.to_csv('ecg_clean.csv', index=False)
print("✅ ECG data saved as ecg_clean.csv")
