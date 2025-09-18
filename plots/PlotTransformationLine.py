# Written with chat-gpt
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Toggles: set to False to disable a curve ---
plot_flags = {
    "main": True,
    "uniform": True,
    "density": True,
}

Y_ = 32
Z_ = 32

# --- Helper function to load and extract transformation slice ---
def load_transformation(filename, y_fixed=Y_, z_fixed=Z_):
    data_vec = np.loadtxt(os.path.join(script_dir, filename), delimiter=",", skiprows=1)
    x, y, z = data_vec[:,0], data_vec[:,1], data_vec[:,2]
    u = data_vec[:,3]

    # Select column where y and z are fixed
    mask = (y == y_fixed) & (z == z_fixed)
    x_col = x[mask]
    u_col = u[mask]

    # Sort by x
    sort_idx = np.argsort(x_col)
    return x_col[sort_idx] - 0.5, u_col[sort_idx]

# --- Main ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load transformations (conditionally)
if plot_flags["main"]:
    x_col_main, u_col_main = load_transformation("transformation.csv")
if plot_flags["uniform"]:
    x_col_uniform, u_col_uniform = load_transformation("uniformTransformationVector.csv")
if plot_flags["density"]:
    x_col_density, u_col_density = load_transformation("densityTransformationVector.csv")

# --- Load histogram ---
data_hist = np.loadtxt(os.path.join(script_dir, "histogram.csv"), delimiter=",", skiprows=1)
xh, yh, zh, counts = data_hist[:,0], data_hist[:,1], data_hist[:,2], data_hist[:,3]

mask_hist = (yh == Y_) & (zh == Z_)
xh_col = xh[mask_hist]
counts_col = counts[mask_hist]

# Sort by x
sort_idx_h = np.argsort(xh_col)
xh_col = xh_col[sort_idx_h]
counts_col = counts_col[sort_idx_h]

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(12,8))

# Plot transformations
if plot_flags["main"]:
    ax1.plot(x_col_main, u_col_main, color="blue", marker=".", label="Transformation")
if plot_flags["uniform"]:
    ax1.plot(x_col_uniform, u_col_uniform, color="green", marker=".", label="Uniform Transformation")
if plot_flags["density"]:
    ax1.plot(x_col_density, u_col_density, color="red", marker=".", label="Density Transformation")

# Add horizontal line at 0
ax1.axhline(0, color="black", linewidth=1, linestyle=":")

ax1.set_xlabel("X coordinate")
ax1.set_ylabel("Transformation along X coordinate")
ax1.tick_params(axis="y")

# Histogram on right y-axis
ax2 = ax1.twinx()
ax2.bar(xh_col, counts_col, alpha=0.3, color="orange", label="Histogram Count")
ax2.set_ylabel("Count", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper right")

plt.title("Transformations vs Histogram along X-axis for Y=" + str(Y_) + " and Z=" + str(Z_))
fig.tight_layout()
plt.show()
