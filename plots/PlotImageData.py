# generated with chat-gpt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.join(script_dir, "image.csv"), delimiter=",", skiprows=1)

r, g, b = data[:,0], data[:,1], data[:,2]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection="3d")

# Plot points in RGB color
ax.scatter(r, g, b, c=data, marker=".", s=10, alpha=0.6)

# Axes setup
ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
#ax.set_title("RGB Point Cloud")
ax.set_aspect("equal")
ax.view_init(15, -25, 0)
plt.savefig("fjord_pointcloud.png", dpi=200, bbox_inches="tight")
plt.show()
