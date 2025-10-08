#written wight chat-gpt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap

umap_df = pd.read_csv("../bin/dataOutput.csv")
X = umap_df[["X", "Y", "Z"]].values
y = umap_df["label"].values


classes = np.unique(y)
num_classes = len(classes)

cmap = get_cmap("tab10", num_classes)  # tab10 has 10 distinct colors
norm = BoundaryNorm(np.arange(classes.min() - 0.5, classes.max() + 1.5), cmap.N)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X[:, 0], X[:, 1], X[:, 2],
    c=y,
    cmap=cmap,
    norm=norm,
    s=5,
    alpha=0.7
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_aspect("equal")
ax.view_init(15, 30, 0)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)


cbar = fig.colorbar(scatter, ax=ax, ticks=classes)
cbar.ax.set_yticklabels(classes)
cbar.set_label("Label")

plt.savefig("digitsplot_20.png", dpi=150, bbox_inches="tight")
plt.show()
