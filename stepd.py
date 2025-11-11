import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rr_df = pd.read_csv('ecg_rr_hr.csv')
rr = rr_df['rr_s'].values

plt.figure(figsize=(10,3))
plt.plot(rr, marker='o')
plt.xlabel('Beat index')
plt.ylabel('RR interval (s)')
plt.title('RR Tachogram')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,3))
plt.hist(rr, bins=8)
plt.xlabel('RR interval (s)')
plt.title('RR Interval Histogram')
plt.tight_layout()
plt.show()
