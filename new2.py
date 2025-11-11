import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load processed ECG and peaks
ecg = pd.read_csv('ecg_processed.csv')
peaks = pd.read_csv('ecg_peaks.csv')

t = ecg['time'].values
v = ecg['voltage'].values
peak_t = peaks['peak_time'].values
peak_v = peaks['peak_voltage'].values

plt.figure(figsize=(12,3))
plt.plot(t, v, label='processed')
plt.scatter(peak_t, peak_v, color='red', s=35, zorder=3, label='detected peaks')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.title('Processed ECG with detected peaks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
