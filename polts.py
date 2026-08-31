import numpy as np
import matplotlib.pyplot as plt

flux = np.loadtxt("data_dynamic/ideal_flux.csv", delimiter=",")
flux_ideal = np.loadtxt("data_dynamic/ideal_flux.csv", delimiter=",")
print(np.shape(flux_ideal))

plt.imshow(flux, cmap='viridis', interpolation='nearest', aspect='equal', extent=[0, 50, 0, 50])
plt.colorbar(label='Flux [kW/m²]')
plt.title("Raytraced Flux Profile")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
