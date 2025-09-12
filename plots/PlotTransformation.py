#written with chat-gpt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Load vector field CSV
# Format: x,y,z,u,v,w
script_dir = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.join(script_dir, "transformation.csv"), delimiter=",", skiprows=1)
# Subsample before plotting
x, y, z = data[:,0], data[:,1], data[:,2]
u, v, w = data[:,3], data[:,4], data[:,5]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")

# Plot all vectors (no stride, since you already subsampled in C++)
ax.quiver(x, y, z, u, v, w,
          length=1.0,   # acts as a global scaling factor
          normalize=False, 
          color="blue")

ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")
ax.set_title("3D Vector Field")

ax.set_aspect("equal")

plt.show()