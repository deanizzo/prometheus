import numpy as np
import pandas as pd

# Field Variables
receiver_diameter = 17.65  # Diameter of the receiver in meters
receiver_height = 21.6  # Height of the receiver in meters
tower_height = 200  # Height of the tower in meters
rated_power = 565  # Rated power of the system in Megawatts
time_of_day = 12  # Time of day in hours (0-23)

# Simulation Variables
min_rays = 100000
x_res = 50
y_res = 50

# CSV Variables
ideal_flux_csv = "ideal_flux.csv"
dynamic_csv = "heliostat_dyn.csv"

receiver_radius = receiver_diameter / 2  # Radius of the receiver in meters
receiver_circumference = 2 * np.pi * receiver_radius  # Circumference of the receiver in meters

# Simulate
# --- Setup CoPylot ---
from copylot import CoPylot
cp = CoPylot(debug=False)
r = cp.data_create()
cp.data_set_string(r, "ambient.0.weather_file", "./climate_files/USA CA Daggett (TMY2).csv")
cp.data_set_number(r, "fluxsim.0.x_res", x_res)
cp.data_set_number(r, "fluxsim.0.y_res", y_res)
cp.data_set_number(r, "fluxsim.0.flux_hour", time_of_day)
cp.data_set_number(r, "solarfield.0.q_des", rated_power)
cp.data_set_number(r, "solarfield.0.tht", tower_height)
cp.data_set_number(r, "receiver.0.rec_diameter", receiver_diameter)
cp.data_set_number(r, "receiver.0.rec_height", receiver_height)
cp.generate_layout(r)
cp.data_set_number(r, "fluxsim.0.min_rays", min_rays)
#cp.data_set_string(r, "fluxsim.0.aim_method", "Simple aim points")
#cp.data_set_string(r, "ambient.0.sun_type", "Point sun")
cp.update_geometry(r)
cp.simulate(r)

results = cp.detail_results(r)


def extract_heliostat_data(cp, r, results):
    """Extract heliostat positions and image sizes."""
    x_locations = np.array(results['x_location'])
    y_locations = np.array(results['y_location'])
    num_heliostats = len(x_locations)
    
    # Calculate geometric properties
    distances = np.sqrt(x_locations**2 + y_locations**2)
    alphas = np.arctan(x_locations / tower_height)  # Elevation angle to tower
    y_scale = np.sqrt(distances**2 + tower_height**2) / distances  # Scale factor for y-dimension of image size
    
    # Get image sizes for all heliostats
    image_sizes = []
    flux = []
    for i in range(num_heliostats):
        image_sizes.append(cp.get_heliostat_image_sizes(r, heliostat_id=i))
        flux.append(np.array(cp.get_heliostat_fluxmap(r, heliostat_id=i)))
    amplitudes = []
    for i in range(num_heliostats):
        amplitudes.append(flux[i].max()*1.1)
    amplitudes = np.array(amplitudes)
    print(f"Amplitudes (first 5): {flux[:5]}")
    # Scale y-dimension of image sizes
    image_sizes = np.array([(tower_height*x, tower_height*y * scale) for (x, y), scale in zip(image_sizes, y_scale)])
    x_normal = (np.arctan2(np.array(y_locations), np.array(x_locations)) - np.pi/2)*receiver_radius
    x_normal = np.mod(x_normal, receiver_circumference)
    y_normal = [receiver_height/2]*len(x_locations)
    
    return x_locations, y_locations, image_sizes, alphas, amplitudes, x_normal, y_normal

def save_all_data(image_sizes, alphas, amplitudes, x_normal, y_normal, filename='./data_dynamic/heliostat_dyn.csv'):
    """Calculate and save cluster averages to CSV."""
    cluster_data = []
    
    for i in range(len(image_sizes)):

        if x_normal[i] < np.pi*receiver_diameter/2 and x_normal[i] > 0:
            xi = x_normal[i] + np.pi*receiver_diameter/2
        else:
            xi = x_normal[i] - np.pi*receiver_diameter/2

        cluster_data.append({
            'heliostat_id': i,
            'sigma_x': image_sizes[i][0],
            'sigma_y': image_sizes[i][1],
            #'alpha': alphas[i],
            #'alpha_degrees': np.degrees(alphas[i]),
            'amplitude': amplitudes[i],
            'x_locations': x_locations[i],
            'y_locations': y_locations[i],
            'x_normal': xi,
            'y_normal': y_normal[i]
        })
    
    df = pd.DataFrame(cluster_data)
    df.to_csv(filename, index=False)
    print(f"\nCluster averages saved to '{filename}'")
    
    return df

x_locations, y_locations, image_sizes, alphas, amplitudes, x_normal, y_normal = extract_heliostat_data(cp, r, results)
save_all_data(image_sizes, alphas, amplitudes, x_normal, y_normal)

assert cp.data_free(r)

