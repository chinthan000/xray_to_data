import numpy as np
import pandas as pd

ecg = pd.read_csv('ecg_processed.csv')
t = ecg['time'].values
v = ecg['voltage'].values

# Remove DC offset using median (robust to outliers)
v_centered = v - np.median(v)

# Optional: scale so QRS peaks sit in expected amplitude range (skip unless needed)
# v_scaled = v_centered / np.max(np.abs(v_centered)) * desired_peak_mv

# Save new centered signal
out = pd.DataFrame({'time': t, 'voltage': v_centered})
out.to_csv('ecg_processed_centered.csv', index=False)
print("Saved ecg_processed_centered.csv")
