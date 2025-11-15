#written with chat-gpt
import os
import numpy as np
import plotly.graph_objects as go

# ======= Hardcode the list of CSV files here =======
csv_files = [
    "histogram.csv",
    "integral0.csv",
    #"integral1.csv",
    #"integral2.csv",
    #"integral3.csv",
    #"integral4.csv",
    #"integral5.csv",
    #"integral6.csv",
    #"integral7.csv",
    #"integral8.csv",
    #"integral9.csv",
    #"integral10.csv",
    #"integral11.csv",
    #"integral12.csv",
    #"integral13.csv",
]
# Directory where your Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Parameters
binsR = binsG = binsB = 32 # must match the binCount set in the config.yaml file!!!

for csv_name in csv_files:
    csv_path = os.path.join(script_dir, csv_name)
    
    # Load CSV
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=float)
    r = data[:, 0].astype(int)
    g = data[:, 1].astype(int)
    b = data[:, 2].astype(int)
    counts = data[:, 3]
    
    # Create 3D histogram array
    hist = np.zeros((binsR, binsG, binsB), dtype=float)
    for rr, gg, bb, c in zip(r, g, b, counts):
        hist[rr, gg, bb] = c
    
    # Plotly Volume
    fig = go.Figure(data=go.Volume(
        x=np.repeat(np.arange(binsR), binsG * binsB),
        y=np.tile(np.repeat(np.arange(binsG), binsB), binsR),
        z=np.tile(np.arange(binsB), binsR * binsG),
        value=hist.flatten(),
        opacity=0.05,
        surface_count=100
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="R",
            yaxis_title="G",
            zaxis_title="B"
        ),
        title=f"3D RGB Histogram Density ({csv_name})"
    )
    
    fig.show()