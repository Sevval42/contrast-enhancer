# generated with chat-gpt
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.join(script_dir, "standardDeviation.csv"), delimiter=",", skiprows=1)

iteration, x = data[:,0], data[:,1]

plt.figure(figsize=(8,5))
plt.plot(iteration, x, linestyle="-", color="blue")

plt.xlabel("Iteration")
plt.ylabel("standard deviation")
plt.yscale("log") # for better visibility of gradient
plt.title("Standard deviation over Iterations")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
