import numpy as np
import matplotlib.pyplot as plt

for i in range(40):
    flux = np.loadtxt(f"outputs/mma/data/flux_map_iter_{i+1}.csv", delimiter=",")
    plt.imshow(flux, cmap='viridis', interpolation='nearest', aspect='equal', extent=[0, 50, 0, 50], vmin=0, vmax=900)
    plt.colorbar(label='Flux [kW/m²]')
    plt.title(f"Flux Optimization Evolution - Iteration {i+1}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"outputs/mma/plots/flux_iter_0{i+1:02d}.png", dpi=300)
    plt.close()