import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

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