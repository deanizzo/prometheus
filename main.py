import matplotlib.pyplot as plt
from copylot import CoPylot
import numpy as np

## Minimum working example -> Must update path to weather file
cp = CoPylot()
r = cp.data_create()
assert cp.data_set_string(
        r,
        "ambient.0.weather_file",
        "./climate_files/USA CA Daggett (TMY2).csv",
    )
cp.data_set_string(r, "fluxsim.0.aim_method", "Image size priority")
cp.data_set_number(r, "fluxsim.0.x_res", 54)
cp.data_set_number(r, "fluxsim.0.y_res", 54)
cp.data_set_number(r, "solarfield.0.q_des", 220)
cp.data_set_number(r, "solarfield.0.tht", 170)
cp.data_set_number(r, "fluxsim.0.flux_hour", 11.82)
assert cp.generate_layout(r)
field = cp.get_layout_info(r)
assert cp.simulate(r)
flux = cp.get_fluxmap(r)
assert cp.data_free(r)

# Plotting (default) solar field and flux map
# Solar Field
plt.scatter(field['x_location'], field['y_location'], s=1.5)
plt.tight_layout()
plt.show()
flux = np.array(flux)
print(np.shape(flux))
# flux
np.savetxt(f'./outputs/flux_map_{11.82}_{220}_{17.65}x{21.6}_Simple_aim_points.csv', flux, delimiter=",", fmt="%.6f")
im = plt.imshow(flux)
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f'./outputs/flux_map_{11.82}_{220}_{17.65}x{21.6}_Simple_aim_points.png')
plt.show()