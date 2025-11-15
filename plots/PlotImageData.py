import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input and output folders
input_dir = os.path.join(script_dir, "imageData")
output_dir = os.path.join(script_dir, "imageDataOutput")

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop over all CSV files in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Processing {filename} → {output_filename}")

        # Load CSV data (skip header row)
        data = np.loadtxt(input_path, delimiter=",", skiprows=1)

        # Expecting columns: R, G, B
        r, g, b = data[:, 0], data[:, 1], data[:, 2]

        # Create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot RGB scatter
        ax.scatter(r, g, b, c=data, marker=".", s=10, alpha=0.6)

        # Axes setup
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_aspect("equal")
        ax.view_init(15, -25, 0)
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)  # Close to free memory

print("✅ All CSV files processed and plots saved to /imageDataOutput")
