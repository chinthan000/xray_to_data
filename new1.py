# Step 1: Plot raw signal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv('ecg_clean.csv')   # replace filename if different
time = df['time'].values
voltage = df['voltage'].values

# Quick info
print("Samples:", len(time))
print("Time range: {:.3f} to {:.3f} s".format(time.min(), time.max()))
dt = np.median(np.diff(time))
fs = 1.0/dt if dt>0 else None
print("Estimated sampling interval dt = {:.6f} s, fs ≈ {:.1f} Hz".format(dt, fs))

# Plot
plt.figure(figsize=(12,3))
plt.plot(time, voltage, linewidth=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.title('Raw extracted ECG')
plt.grid(True)
plt.tight_layout()
plt.show()



# Step 2: Interpolate onto uniform time grid
from scipy.interpolate import interp1d

# target sampling frequency (choose near original estimate). If fs is None or weird, use 250 Hz.
target_fs = int(round(fs)) if fs and fs>0 else 250
target_fs = max(200, min(target_fs, 1000))  # keep in reasonable ECG range
print("Using target sampling rate:", target_fs, "Hz")

t_uniform = np.arange(time.min(), time.max(), 1.0/target_fs)
interp = interp1d(time, voltage, kind='linear', fill_value='extrapolate')
v_uniform = interp(t_uniform)

# Plot resampled
plt.figure(figsize=(12,3))
plt.plot(t_uniform, v_uniform, linewidth=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.title(f'Resampled ECG at {target_fs} Hz')
plt.grid(True)
plt.tight_layout()
plt.show()


# Step 3: Baseline removal + smoothing
from scipy.signal import butter, filtfilt, savgol_filter

def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5*fs
    b,a = butter(order, cutoff/nyq, btype='high', analog=False)
    return b,a

# High-pass to remove slow baseline wander (cutoff around 0.5 Hz)
hp_cutoff = 0.5  # Hz, adjust 0.3-0.8 if needed
b,a = butter_highpass(hp_cutoff, target_fs, order=2)
v_hp = filtfilt(b,a,v_uniform)

# Smooth small high-frequency noise with Savitzky-Golay
# window length must be odd and less than signal length
win = 51 if len(v_hp) > 200 else (len(v_hp)//2)*2+1
v_smooth = savgol_filter(v_hp, window_length=win, polyorder=3)

# Plot before/after
plt.figure(figsize=(12,4))
plt.plot(t_uniform, v_uniform, label='raw', alpha=0.4)
plt.plot(t_uniform, v_smooth, label='hp + smooth', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.title('Baseline removed and smoothed ECG')
plt.grid(True)
plt.tight_layout()
plt.show()


# Step 4: Simple R-peak detection using find_peaks
from scipy.signal import find_peaks

signal = v_smooth
# typical adult HR: 40-200 bpm -> 0.3-3.3 Hz. Convert to samples:
min_hr = 40; max_hr = 200
min_dist_sec = 60.0 / max_hr   # min distance between beats seconds
min_dist_samples = int(0.5 * target_fs)  # safe default ~0.5s

# height threshold: median + k*std
height_thresh = np.median(signal) + 0.5*np.std(signal)

peaks, props = find_peaks(signal, distance=min_dist_samples, height=height_thresh, prominence=0.3)
print("Detected peaks:", len(peaks))

# Plot with detected peaks
plt.figure(figsize=(12,3))
plt.plot(t_uniform, signal, label='processed')
plt.plot(t_uniform[peaks], signal[peaks], 'ro', label='R peaks')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.title('Detected R-peaks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute instantaneous heart rate
rr_intervals = np.diff(t_uniform[peaks])  # seconds
if len(rr_intervals)>0:
    hr_inst = 60.0 / rr_intervals
    print("Mean HR (bpm):", np.mean(hr_inst))
    print("Median HR (bpm):", np.median(hr_inst))
else:
    print("Not enough peaks to compute HR.")
# Step 5: Save cleaned results
out_df = pd.DataFrame({'time': t_uniform, 'voltage': signal})
out_df.to_csv('ecg_processed.csv', index=False)
peaks_df = pd.DataFrame({
    'peak_time': t_uniform[peaks],
    'peak_voltage': signal[peaks]
})
peaks_df.to_csv('ecg_peaks.csv', index=False)
print("Saved: ecg_processed.csv and ecg_peaks.csv")
