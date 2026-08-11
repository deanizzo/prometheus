import numpy as np
import matplotlib.pyplot as plt

flux = np.loadtxt("outputs/flux_map_16_550_17.65x21.6.csv", delimiter=",")
flux_ideal = np.loadtxt("data_dynamic/ideal_flux.csv", delimiter=",")
print(np.shape(flux_ideal))

plt.imshow(flux, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Flux [kW/m²]')
plt.title("Raytraced Flux Profile")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
