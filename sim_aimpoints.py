import matplotlib.pyplot as plt
from copylot import CoPylot
import numpy as np
from numpy.polynomial.hermite_e import hermeval
from scipy.optimize import minimize
import os
import numpy as np
import pandas as pd
import random

receiver_diameter = 17.65  # Diameter of the receiver in meters
receiver_height = 21.6  # Height of the receiver in meters
tower_height = 200  # Height of the tower in meters
rated_power = 565  # Rated power of the system in Megawatts
x_res = 48
y_res = 61
time_of_day = 12  # Time of day in hours (0-23)
save_flux_map = True  # Save flux map as PNG
    
# Import Aimpoints
cyl_radius = receiver_diameter / 2
cyl_tht = tower_height
cyl_height = receiver_height

cyl_circumference_half = cyl_radius*np.pi
#script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = "./data_dynamic/heliostat_dyn.csv"
df = pd.read_csv(csv_path)
x_n = df['x_normal'].tolist()
y_n = df['y_normal'].tolist()
#x_n = x_normal
#y_n = y_normal
#a_opt = df['a_opt'].tolist()
#b_opt = df['b_opt'].tolist()
a_opt = df['a_opt'].tolist()
b_opt = df['b_opt'].tolist()
ids_csv = df['heliostat_id'].tolist()
aimpoint_cyl = [
    (
        x + (a - 0.5) * cyl_circumference_half,
        y + (0 - b) * cyl_height
    )
    for x, y, a, b in zip(x_n, y_n, a_opt, b_opt)
]

# Convert aimpoints to global coordinates

aimpoint_cart_x = [cyl_radius * np.sin(aimpoint_cyl[i][0]/cyl_radius) for i in range(len(aimpoint_cyl))]
aimpoint_cart_y = [cyl_radius * -np.cos(aimpoint_cyl[i][0]/cyl_radius) for i in range(len(aimpoint_cyl))]
aimpoint_cart_z = [aimpoint_cyl[i][1] + cyl_tht for i in range(len(aimpoint_cyl))]
# --- Simulation Variables ---
thermal_power_rating = rated_power  # MWth

# --- Setup CoPylot ---
cp = CoPylot(debug=False)
r = cp.data_create()
cp.data_set_string(r, "ambient.0.weather_file", "./climate_files/USA CA Daggett (TMY2).csv")
cp.data_set_number(r, "fluxsim.0.x_res", x_res)
cp.data_set_number(r, "fluxsim.0.y_res", y_res)
cp.data_set_number(r, "solarfield.0.q_des", thermal_power_rating)
cp.data_set_number(r, "solarfield.0.tht", tower_height)
cp.data_set_number(r, "fluxsim.0.flux_hour", time_of_day)
cp.generate_layout(r)
#cp.data_set_string(r, "fluxsim.0.flux_model", "SolTrace")
#cp.data_set_number(r, "fluxsim.0.min_rays", 1000000)
#cp.data_set_number(r, "fluxsim.0.max_rays", 100000000)
#cp.data_set_string(r, "heliostat.0.focus_method", "Flat")
cp.data_set_string(r, "ambient.0.sun_type", "Point sun")
cp.update_geometry(r)

# --- Simulate ---
helio_id = 0
cp.simulate(r, nthreads=12)
results = cp.detail_results(r)
ids = results["id"]
id_list = []
id_list = results["id"].values.tolist()
#random.shuffle(id_list)
#helio_dict = {"id": id_list, "enabled": [0]*helio_id + [1]+[1] * (len(id_list)-helio_id-1)}
aim_dict = {"id": id_list, 'aimpoint-x': aimpoint_cart_x, 'aimpoint-y': aimpoint_cart_y, 'aimpoint-z': aimpoint_cart_z}

cp.data_set_string(r, "fluxsim.0.aim_method", "Keep existing")
#cp.modify_heliostats(r, helio_dict)
cp.modify_heliostats(r, aim_dict)
cp.simulate(r)

flux = cp.get_fluxmap(r)
#print(helio_flux)
#assert cp.data_free(r)

if save_flux_map:
    np.savetxt(f'./outputs/flux_map_{time_of_day}_{rated_power}_{receiver_diameter}x{receiver_height}.csv', flux, delimiter=",", fmt="%.6f")
    plt.imshow(flux, cmap='viridis', interpolation='nearest', extent=[0, cyl_circumference_half*2, 0, cyl_height])
    plt.savefig(f'./outputs/flux_map_{time_of_day}_{rated_power}_{receiver_diameter}x{receiver_height}.png')
else:
    plt.imshow(flux, cmap='viridis', interpolation='nearest', extent=[0, cyl_circumference_half*2, 0, cyl_height])
plt.colorbar(label='Flux [kW/m²]')
plt.title("Raytraced Flux Profile")
plt.xlabel("x")
plt.ylabel("y")
plt.show()