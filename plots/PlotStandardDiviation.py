import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.join(script_dir, "standardDeviation.csv"), delimiter=",", skiprows=1)

iteration, x = data[:,0], data[:,1]

# Find first increase compared to previous value
increase_idx = None
for i in range(1, len(x)):
    if x[i] > x[i-1]:
        increase_idx = i
        break

plt.figure(figsize=(8,5))
plt.plot(iteration, x, linestyle="-", color="blue")

plt.xlabel("Iteration")
plt.ylabel("mse to perfect uniformity")
plt.yscale("log")  # log scale
plt.grid(True, linestyle="--", alpha=0.5)

# Draw vertical red dotted line if increase was found
if increase_idx is not None:
    plt.axvline(x=iteration[increase_idx], color="red", linestyle=":", label="First increase")
    plt.legend()

plt.tight_layout()
plt.savefig("variance_1024_fjord.png", dpi=200)
plt.show()
