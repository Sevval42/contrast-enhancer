import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Files and labels
files = {
    "standardDeviation_0_1.csv": "0.1",
    "standardDeviation_0_5.csv": "0.5",
    "standardDeviation_1.csv": "1",
    "standardDeviation_2.csv": "2",
    "standardDeviation_10.csv": "10"
}

plt.figure(figsize=(8,5))

for fname, label in files.items():
    path = os.path.join(script_dir, fname)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = data[:11]
    iteration, x = data[:,0], data[:,1]
    plt.plot(iteration, x, label=f"p={label}")

plt.xlabel("Iteration")
plt.ylabel("MSE to perfect uniformity")
#plt.yscale("log")  # since you used log scale before
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="$h_0$ factor")
plt.tight_layout()
plt.savefig("mse_multiple_h0.png", dpi=150)
plt.show()
