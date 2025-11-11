import pandas as pd
import numpy as np

peaks = pd.read_csv('ecg_peaks.csv')
peak_t = np.array(peaks['peak_time'])

# RR intervals (s)
rr = np.diff(peak_t)
# instantaneous HR (bpm)
hr_inst = 60.0 / rr

# Basic HRV/time-domain metrics
mean_rr = np.mean(rr)
sdnn = np.std(rr, ddof=1)           # SD of RR (s)
rmssd = np.sqrt(np.mean(np.diff(rr)**2))  # root mean square of successive diffs
mean_hr = np.mean(hr_inst)

print(f"Beats detected: {len(peak_t)}")
print(f"Mean HR (bpm): {mean_hr:.1f}")
print(f"Mean RR (s): {mean_rr:.3f}")
print(f"SDNN (s): {sdnn:.4f}")
print(f"RMSSD (s): {rmssd:.4f}")

# Save RR and HR
pd.DataFrame({'rr_s': rr, 'hr_bpm': hr_inst}).to_csv('ecg_rr_hr.csv', index=False)
print("Saved ecg_rr_hr.csv")
