# Written with chat-gpt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
# Load vector field CSV
# Format: x,y,z,u,v,w
script_dir = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.join(script_dir, "transformation.csv"), delimiter=",", skiprows=1)


data_vec = np.loadtxt(os.path.join(script_dir, "transformation.csv"), delimiter=",", skiprows=1)
x, y, z = data_vec[:,0], data_vec[:,1], data_vec[:,2]
u, v, w = data_vec[:,3], data_vec[:,4], data_vec[:,5]

# Select column: x varies, y=0, z=0
mask = (y == 32) & (z == 32)
x_col = x[mask]
u_col = u[mask]

# Sort by x
sort_idx = np.argsort(x_col)
x_col = x_col[sort_idx]
u_col = u_col[sort_idx]

# --- Load histogram ---
data_hist = np.loadtxt(os.path.join(script_dir, "histogram.csv"), delimiter=",", skiprows=1)
xh, yh, zh, counts = data_hist[:,0], data_hist[:,1], data_hist[:,2], data_hist[:,3]

# Select same axis: y=0, z=0
mask_hist = (yh == 32) & (zh == 32)
xh_col = xh[mask_hist]
counts_col = counts[mask_hist]

# Sort by x
sort_idx_h = np.argsort(xh_col)
xh_col = xh_col[sort_idx_h]
counts_col = counts_col[sort_idx_h]

# --- Plot both on twin axes ---
fig, ax1 = plt.subplots(figsize=(8,5))

# Transformation line
ax1.plot(x_col-0.5, u_col, marker=".", color="blue", label="U (Transformation)")
ax1.set_xlabel("X coordinate")
ax1.set_ylabel("Transformation along X coordinate", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Histogram on right y-axis
ax2 = ax1.twinx()
ax2.bar(xh_col, counts_col, alpha=0.3, color="orange", label="Histogram Count")
ax2.set_ylabel("Count", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

plt.title("Transformation vs Histogram along X-axis")
fig.tight_layout()
plt.show()